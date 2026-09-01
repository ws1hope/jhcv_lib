#include "JHDeepCore.h"
#include "guokuache_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class GuokuacheServicePrivate {
public:
    explicit GuokuacheServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadGuokuacheConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::GuokuachePipeline>(config_);
    }

    const GuokuacheServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Guokuache.txt";
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
            picture_path = req_json.value("picture_path", "");
            if (req_json.contains("station_id")) {
                const auto& v = req_json["station_id"];
                station_id = v.is_string() ? std::stoi(v.get<std::string>()) : v.get<int>();
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path << std::endl;

        // 单图：取 '#' 分隔后的第一个路径
        std::vector<std::string> path_array = FileHelper::splitStringByCsharp(picture_path);
        std::string first_path = path_array.empty() ? picture_path : path_array[0];

        json root = buildResult(first_path, station_id, false, fout);

        std::cout << "[RESULT] " << root.dump() << std::endl;
        fout.close();

        return root.dump();
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

        std::vector<std::string> path_array = FileHelper::splitStringByCsharp(image_path);
        std::string first_path = path_array.empty() ? image_path : path_array[0];

        auto start = std::chrono::high_resolution_clock::now();
        std::ofstream fout;  // 本地测试不写日志
        json root = buildResult(first_path, station_id, true, fout);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

private:
    // 单图处理，返回 zbhc 式数组格式：
    //   all_results 为数组，其每个元素包含该图的识别结果：
    //     read_picture_flag - 图像解析是否成功 OK/NG
    //     rec_state_flag    - 识别是否成功 OK/NG
    //     rec_results       - 各目标 OCR 文本数组（无结果时为空数组）
    //     picture_path      - 结果图保存路径
    json buildResult(const std::string& picture_path,
                     int station_id, bool verbose,
                     std::ofstream& fout)
    {
        json root;
        root["station_id"] = std::to_string(station_id);
        json all_results = json::array();
        json item;

        cv::Mat src_img = cv::imread(picture_path);
        if (!src_img.data) {
            item["read_picture_flag"] = "NG";
            item["rec_state_flag"] = "NG";
            item["rec_results"] = json::array();
            item["picture_path"] = picture_path;
            all_results.push_back(item);
            root["all_results"] = all_results;
            fout << "detect failed! empty image: " << picture_path << std::endl;
            return root;
        }

        Pipeline::GuokuachePipelineResult pr = pipeline_->process(src_img, verbose);

        // 收集非空 OCR 文本（按 Z 型顺序）
        std::vector<std::string> texts;
        for (const auto& t : pr.targets) {
            if (!t.ocr_text.empty()) texts.push_back(t.ocr_text);
        }

        // 结果图保存路径：本地测试用当前目录，服务端用 result_dir
        std::string save_path;
        if (verbose) {
            save_path = "guokuache_result.jpg";
        } else {
            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);
            std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            FileHelper::ensureDirectoryExists(folderPath);

            save_path = cv::format(
                "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d.jpg",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec);
        }

        if (!pr.annotated_image.empty()) {
            cv::imwrite(save_path, pr.annotated_image);
        } else {
            cv::imwrite(save_path, src_img);
        }

        item["read_picture_flag"] = "OK";
        item["rec_state_flag"] = texts.empty() ? "NG" : "OK";
        item["rec_results"] = texts;
        item["picture_path"] = save_path;
        all_results.push_back(item);
        root["all_results"] = all_results;

        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }
        return root;
    }

    GuokuacheServerConfig config_;
    std::unique_ptr<Pipeline::GuokuachePipeline> pipeline_;
    std::mutex mtx_;
};

GuokuacheService::GuokuacheService(const std::string& config_path)
    : m_pHandle(std::make_shared<GuokuacheServicePrivate>(config_path))
{
}

GuokuacheService::~GuokuacheService() = default;

const GuokuacheServerConfig& GuokuacheService::config() const
{
    return m_pHandle->config();
}

std::string GuokuacheService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int GuokuacheService::runLocalTest(const std::string& image_path,
                                   const std::string& heat_number,
                                   int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
