#include "JHDeepCore.h"
#include "zbhc_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

// 单图置信度 = 所有结果(坯料, ocr_text 非空)置信度的最小值；无结果返回 0
static float computeImageConfidence(const Pipeline::ZbhcPipelineResult& r)
{
    float img_min = 1.0f;
    bool any = false;
    for (auto& billet : r.billets) {
        if (billet.ocr_text.empty()) continue;
        img_min = std::min(img_min, billet.ocr_confidence);
        any = true;
    }
    return any ? img_min : 0.0f;
}

// 多图择优：只保留图置信度最大的那张图（confs 非空且与 array_result 等长时调用）
static size_t keepBestImageConfidence(json& array_result,
                                      const std::vector<float>& confs)
{
    size_t best = 0;
    for (size_t i = 1; i < confs.size(); i++) {
        if (confs[i] > confs[best]) best = i;
    }
    json best_item = array_result[best];
    array_result = json::array();
    array_result.push_back(std::move(best_item));
    return best;
}

class ZbhcServicePrivate {
public:
    explicit ZbhcServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadZbhcConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::ZbhcPipeline>(config_);
    }

    const ZbhcServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Zbhc.txt";
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
        std::vector<float> img_confs;
        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;

            if (!src_img.data) {
                item["read_picture_flag"] = "NG";
                item["rec_state_flag"] = "NG";
                item["rec_results"] = json::array();
                item["picture_path"] = "";
                array_result.push_back(item);
                img_confs.push_back(0.0f);
                fout << "detect failed! empty image" << std::endl;
                continue;
            }

            Pipeline::ZbhcPipelineResult pipeline_result = pipeline_->process(src_img, false);
            img_confs.push_back(computeImageConfidence(pipeline_result));

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

            if (!pipeline_result.annotated_image.empty()) {
                cv::imwrite(save_picture_name, pipeline_result.annotated_image);
            } else if (!src_img.empty()) {
                cv::imwrite(save_picture_name, src_img);
            }

            saveCharCrops(pipeline_result, pic_number, t);

            json billets_ocr = json::array();
            for (auto& billet : pipeline_result.billets) {
                if (!billet.ocr_text.empty()) {
                    std::string text = billet.ocr_text;
                    if (text.size() >= 3) {
                        text.insert(text.size() - 3, "#");
                    }
                    billets_ocr.push_back(text);
                }
            }

            item["read_picture_flag"] = "OK";
            item["rec_state_flag"] = billets_ocr.empty() ? "NG" : "OK";
            item["rec_results"] = billets_ocr;
            item["picture_path"] = save_picture_name;

            array_result.push_back(item);
        }

        if (!img_confs.empty()) {
            size_t best_idx = keepBestImageConfidence(array_result, img_confs);
            std::cout << "[BEST] idx=" << best_idx
                      << " conf=" << img_confs[best_idx]
                      << " kept=" << array_result.size() << "/" << img_confs.size() << std::endl;
        }

        json root_all;
        root_all["station_id"] = std::to_string(station_id);
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
        std::vector<float> img_confs;

        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;

            if (!src_img.data) {
                std::cerr << "[ERROR] Cannot read image: " << picture_path_array[pic_number] << std::endl;
                item["read_picture_flag"] = "NG";
                item["rec_state_flag"] = "NG";
                item["rec_results"] = json::array();
                item["picture_path"] = "";
                array_result.push_back(item);
                img_confs.push_back(0.0f);
                continue;
            }

            Pipeline::ZbhcPipelineResult pipeline_result = pipeline_->process(src_img, true);
            img_confs.push_back(computeImageConfidence(pipeline_result));

            json billets_ocr = json::array();
            for (auto& billet : pipeline_result.billets) {
                if (!billet.ocr_text.empty()) {
                    std::string text = billet.ocr_text;
                    if (text.size() >= 3) {
                        text.insert(text.size() - 3, "#");
                    }
                    billets_ocr.push_back(text);
                }
            }

            item["read_picture_flag"] = "OK";
            item["rec_state_flag"] = billets_ocr.empty() ? "NG" : "OK";
            item["rec_results"] = billets_ocr;
            item["picture_path"] = "";

            array_result.push_back(item);

            if (!pipeline_result.annotated_image.empty()) {
                std::string save_path = "zbhc_result_" + std::to_string(pic_number) + ".jpg";
                cv::imwrite(save_path, pipeline_result.annotated_image);
                std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
            }

            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);
            saveCharCrops(pipeline_result, pic_number, t);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!img_confs.empty()) {
            size_t best_idx = keepBestImageConfidence(array_result, img_confs);
            std::cout << "[BEST] idx=" << best_idx
                      << " conf=" << img_confs[best_idx]
                      << " kept=" << array_result.size() << "/" << img_confs.size() << std::endl;
        }

        json root_all;
        root_all["station_id"] = std::to_string(station_id);
        root_all["all_results"] = array_result;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root_all.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    // 保存每个字符的方向矫正后裁剪图（OCR 输入）到 char_crop_dir，扁平命名：
    // <char_crop_dir>\YYYYMMDDHHMMSS_p<pic>_b<billet>_c<char>.jpg
    // char_crop_dir 为空时跳过
    void saveCharCrops(const Pipeline::ZbhcPipelineResult& result,
                       int pic_number, const tm* t)
    {
        if (config_.char_crop_dir.empty()) return;
        FileHelper::ensureDirectoryExists(config_.char_crop_dir);
        for (int bi = 0; bi < (int)result.billets.size(); bi++) {
            const auto& billet = result.billets[bi];
            for (int ci = 0; ci < (int)billet.chars.size(); ci++) {
                const cv::Mat& crop = billet.chars[ci].image_after_flip;
                if (crop.empty()) continue;
                std::string save_name = cv::format("%s\\%d%02d%02d%02d%02d%02d_p%d_b%d_c%d.jpg",
                    config_.char_crop_dir.c_str(),
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                    t->tm_hour, t->tm_min, t->tm_sec,
                    pic_number, bi + 1, ci + 1);
                cv::imwrite(save_name, crop);
            }
        }
    }

    ZbhcServerConfig config_;
    std::unique_ptr<Pipeline::ZbhcPipeline> pipeline_;
    std::mutex mtx_;
};

ZbhcService::ZbhcService(const std::string& config_path)
    : m_pHandle(std::make_shared<ZbhcServicePrivate>(config_path))
{
}

ZbhcService::~ZbhcService() = default;

const ZbhcServerConfig& ZbhcService::config() const
{
    return m_pHandle->config();
}

std::string ZbhcService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int ZbhcService::runLocalTest(const std::string& image_path,
                               const std::string& heat_number,
                               int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
