#include "JHDeepCore.h"
#include "tiebiao_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class TiebiaoServicePrivate {
public:
    explicit TiebiaoServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadTiebiaoConfig(config_path))
    {
        TiebiaoConfig pcfg;
        pcfg.label_seg_model = config_.label_seg_model;
        pcfg.char_seg_model = config_.char_seg_model;
        pcfg.ocr_model = config_.ocr_model;
        pcfg.ocr_label = config_.ocr_label;
        pcfg.direction_cls_model = config_.direction_cls_model;
        pcfg.device = config_.device;

        pipeline_ = std::make_unique<Pipeline::TiebiaoPipeline>(pcfg);
    }

    const TiebiaoServerConfig& config() const { return config_; }

    TiebiaoResult recognize(const cv::Mat& image,
                                       int station_id,
                                       const std::string& heat_number,
                                       bool verbose)
    {
        return pipeline_->process(image, station_id, heat_number, verbose);
    }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Tiebiao.txt";
        std::string fileFullName = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(fileFullName.c_str(), std::ios::app);

        int station_id = 0;
        std::string heat_number;
        std::string picture_path;

        try {
            json req_json = json::parse(req_body);
            heat_number = req_json.value("heat_number", "");
            station_id = req_json.value("station_id", 0);
            picture_path = req_json.value("picture_path", "");
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path << std::endl;

        std::vector<std::string> picture_path_array = FileHelper::splitStringByCsharp(picture_path);

        json array_result = json::array();
        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;
            item["picture_id"] = pic_number + 1;

            if (!src_img.data) {
                item["state_flag"] = "NG";
                array_result.push_back(item);
                fout << "detect failed! empty image" << std::endl;
                continue;
            }

            TiebiaoResult result = pipeline_->process(
                src_img, station_id, heat_number, false);

            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);

            std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            FileHelper::ensureDirectoryExists(folderPath);

            std::string save_picture_name = cv::format(
                "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_%d.jpg",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec, pic_number);

            if (result.state_flag == "OK" && !result.annotated_image.empty()) {
                cv::imwrite(save_picture_name, result.annotated_image);
            } else if (!src_img.empty()) {
                cv::imwrite(save_picture_name, src_img);
            }

            item["state_flag"] = result.state_flag;
            item["label_type"] = result.label_type;
            item["result"] = result.ocr_text;
            item["picture_path"] = save_picture_name;
            array_result.push_back(item);
        }

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;

        std::cout << "[RESULT] " << root_all.dump() << std::endl;
        fout.close();

        return root_all.dump();
    }

    int runLocalTest(const std::string& image_path,
                      const std::string& heat_number,
                      int station_id)
    {
        std::cout << "=== Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << std::endl;

        std::vector<std::string> picture_path_array = FileHelper::splitStringByCsharp(image_path);

        auto start = std::chrono::high_resolution_clock::now();
        json array_result = json::array();

        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);
            if (!src_img.data) {
                std::cerr << "[ERROR] Cannot read image: " << picture_path_array[pic_number] << std::endl;
                json item;
                item["picture_id"] = pic_number + 1;
                item["state_flag"] = "NG";
                array_result.push_back(item);
                continue;
            }

            TiebiaoResult result = pipeline_->process(
                src_img, station_id, heat_number, true);

            json item;
            item["picture_id"] = pic_number + 1;
            item["state_flag"] = result.state_flag;
            item["label_type"] = result.label_type;
            item["result"] = result.ocr_text;
            array_result.push_back(item);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root_all.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    TiebiaoServerConfig config_;
    std::unique_ptr<Pipeline::TiebiaoPipeline> pipeline_;
    std::mutex mtx_;
};

TiebiaoService::TiebiaoService(const std::string& config_path)
    : m_pHandle(std::make_shared<TiebiaoServicePrivate>(config_path))
{
}

TiebiaoService::~TiebiaoService() = default;

const TiebiaoServerConfig& TiebiaoService::config() const
{
    return m_pHandle->config();
}

std::string TiebiaoService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int TiebiaoService::runLocalTest(const std::string& image_path,
                                  const std::string& heat_number,
                                  int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
