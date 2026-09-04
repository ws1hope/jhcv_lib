#include "JHDeepCore.h"
#include "huaxin_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class HuaxinServicePrivate {
public:
    explicit HuaxinServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadHuaxinConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::HuaxinPipeline>(config_);
    }

    const HuaxinServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Huaxin.txt";
        std::string fileFullName = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(fileFullName.c_str(), std::ios::app);

        std::string station_id = "0";
        std::string heat_number;
        std::string picture_path;

        try {
            json req_json = json::parse(req_body);
            heat_number = req_json.value("heat_number", "");
            picture_path = req_json.value("picture_path", "");
            if (req_json.contains("station_id")) {
                const auto& v = req_json["station_id"];
                if (v.is_string()) {
                    station_id = v.get<std::string>();
                } else if (v.is_number_integer()) {
                    station_id = std::to_string(v.get<int>());
                }
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path << std::endl;

        // 单张图片：picture_path 整体作为图片路径，不按任何字符拆分
        json root;
        root["station_id"] = station_id;
        json all_results = json::array();
        double total_inference_ms = 0.0;
        all_results.push_back(
            buildResult(picture_path, station_id, 1, false, fout, total_inference_ms));
        root["all_results"] = all_results;

        root["time_cost"] = std::round(total_inference_ms * 100.0) / 100.0;

        std::cout << "[RESULT] " << root.dump() << std::endl;
        fout.close();

        return root.dump();
    }

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     const std::string& station_id)
    {
        std::cout << "=== Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << std::endl;

        std::ofstream fout;  // 本地测试不写日志

        json root;
        root["station_id"] = station_id;
        json all_results = json::array();
        double total_inference_ms = 0.0;
        all_results.push_back(
            buildResult(image_path, station_id, 1, true, fout, total_inference_ms));
        root["all_results"] = all_results;

        root["time_cost"] = std::round(total_inference_ms * 100.0) / 100.0;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root.dump(2) << std::endl;
        std::cout << "inference time: " << total_inference_ms << " ms" << std::endl;

        return 0;
    }

private:
    // 单图处理，返回 all_results 数组中的一项：
    //   state_flag  - 图像解析是否成功 OK/NG
    //   detail_state - 识别是否成功 OK/NG（解析失败时为 NG）
    //   result      - 铸坯端面识别结果，失败或无结果为 ""
    //   picture_path - 结果图保存路径（输入图目录\results\station_<id>\<原文件名>）
    json buildResult(const std::string& picture_path,
                     const std::string& station_id,
                     int picture_id, bool verbose,
                     std::ofstream& fout,
                     double& total_inference_ms)
    {
        json item;
        item["picture_id"] = std::to_string(picture_id);

        cv::Mat src_img = cv::imread(picture_path);
        if (!src_img.data) {
            item["state_flag"] = "NG";
            item["detail_state"] = "NG";
            item["result"] = "";
            item["picture_path"] = picture_path;
            fout << "detect failed! empty image: " << picture_path << std::endl;
            return item;
        }

        Pipeline::HuaxinPipelineResult pr = pipeline_->process(src_img, verbose);
        total_inference_ms += pr.inference_time_ms;

        // 结果图保存路径：输入图所在目录\results\station_<工位>\原文件名；
        // 本地测试直接存当前目录
        std::string save_path;
        if (verbose) {
            save_path = "huaxin_result.jpg";
        } else {
            size_t pos = picture_path.find_last_of("\\/");
            std::string dir = (pos == std::string::npos) ? "." : picture_path.substr(0, pos);
            std::string name = (pos == std::string::npos) ? picture_path : picture_path.substr(pos + 1);
            std::string folder = dir + "\\results\\station_" + station_id;
            FileHelper::ensureDirectoryExists(folder);
            save_path = folder + "\\" + name;
        }

        if (!pr.annotated_image.empty()) {
            cv::imwrite(save_path, pr.annotated_image);
        } else {
            cv::imwrite(save_path, src_img);
        }

        item["state_flag"] = "OK";

        if (!pr.all_results.empty()) {
            item["detail_state"] = "OK";
            item["result"] = pr.all_results;
        } else {
            item["detail_state"] = "NG";
            item["result"] = "";
        }

        item["picture_path"] = save_path;

        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }
        return item;
    }

    HuaxinServerConfig config_;
    std::unique_ptr<Pipeline::HuaxinPipeline> pipeline_;
    std::mutex mtx_;
};

HuaxinService::HuaxinService(const std::string& config_path)
    : m_pHandle(std::make_shared<HuaxinServicePrivate>(config_path))
{
}

HuaxinService::~HuaxinService() = default;

const HuaxinServerConfig& HuaxinService::config() const
{
    return m_pHandle->config();
}

std::string HuaxinService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int HuaxinService::runLocalTest(const std::string& image_path,
                                const std::string& heat_number,
                                const std::string& station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
