#include "JHDeepCore.h"
#include "feigangcao_pipeline.h"
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

class FeigangcaoServicePrivate {
public:
    explicit FeigangcaoServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadFeigangcaoConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::FeigangcaoPipeline>(config_);
    }

    const FeigangcaoServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Feigangcao.txt";
        std::string fileFullName = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(fileFullName.c_str(), std::ios::app);

        std::string station_id = "0";
        std::string picture_path;

        try {
            json req_json = json::parse(req_body);
            if (req_json.contains("station_id")) {
                const auto& v = req_json["station_id"];
                if (v.is_string()) {
                    station_id = v.get<std::string>();
                } else if (v.is_number_integer()) {
                    station_id = std::to_string(v.get<int>());
                }
            }
            if (req_json.contains("picture_path")) {
                const auto& v = req_json["picture_path"];
                if (v.is_string()) {
                    picture_path = v.get<std::string>();
                }
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " path:" << picture_path << std::endl;

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
                     const std::string& station_id)
    {
        std::cout << "=== Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
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
    //   state_flag   - 图像解析是否成功 OK/NG
    //   detail_state - 识别是否成功 OK/NG（解析失败时为 NG）
    //   result       - 分类结果（class_name），失败或无结果为 ""
    //   picture_path - 结果图保存路径
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

        Pipeline::FeigangcaoPipelineResult pr = pipeline_->process(src_img, station_id, verbose);
        total_inference_ms += pr.inference_time_ms;

        if (!pr.result.empty()) {
            item["detail_state"] = "OK";
            item["result"] = pr.result;
        } else {
            item["detail_state"] = "NG";
            item["result"] = "";
        }

        // 结果图保存路径：输入图所在目录\results\station_<工位>\<OK|NG>\原文件名；
        // 本地测试直接存当前目录
        std::string save_path;
        if (verbose) {
            save_path = "feigangcao_result.jpg";
        } else {
            size_t pos = picture_path.find_last_of("\\/");
            std::string dir = (pos == std::string::npos) ? "." : picture_path.substr(0, pos);
            std::string name = (pos == std::string::npos) ? picture_path : picture_path.substr(pos + 1);
            std::string folder = dir + "\\results\\station_" + station_id
                                 + "\\" + item["detail_state"].get<std::string>();
            FileHelper::ensureDirectoryExists(folder);
            save_path = folder + "\\" + name;
        }

        if (!pr.annotated_image.empty()) {
            cv::imwrite(save_path, pr.annotated_image);
        } else {
            cv::imwrite(save_path, src_img);
        }

        // 保存定位框裁剪图，以及分类结果图（按类别自动建文件夹）
        if (!pr.det_crop.empty()) {
            size_t pos = picture_path.find_last_of("\\/");
            std::string dir = (pos == std::string::npos) ? "." : picture_path.substr(0, pos);
            std::string name = (pos == std::string::npos) ? picture_path : picture_path.substr(pos + 1);

            // 1) 定位框裁剪图
            std::string det_folder = config_.det_crop_dir.empty()
                ? dir + "\\results\\station_" + station_id + "\\detect"
                : config_.det_crop_dir + "\\station_" + station_id;
            std::string det_path = verbose ? "feigangcao_detect_crop.jpg"
                                           : det_folder + "\\" + name;
            if (!verbose) {
                FileHelper::ensureDirectoryExists(det_folder);
            }
            cv::imwrite(det_path, pr.det_crop);
            if (verbose) {
                std::cout << "[INFO] Detect crop saved: " << det_path << std::endl;
            }
            fout << "detect crop saved: " << det_path << std::endl;

            // 2) 分类结果图，按类别名建子文件夹
            std::string cls_name = pr.result.empty() ? "NG" : pr.result;
            std::string cls_folder = config_.cls_crop_dir.empty()
                ? dir + "\\results\\station_" + station_id + "\\classify\\" + cls_name
                : config_.cls_crop_dir + "\\" + cls_name;
            std::string cls_path = verbose ? "feigangcao_classify_" + cls_name + ".jpg"
                                           : cls_folder + "\\" + name;
            if (!verbose) {
                FileHelper::ensureDirectoryExists(cls_folder);
            }
            cv::imwrite(cls_path, pr.det_crop);
            if (verbose) {
                std::cout << "[INFO] Classify image saved: " << cls_path << std::endl;
            }
            fout << "classify image saved: " << cls_path << std::endl;
        }

        item["state_flag"] = "OK";

        item["picture_path"] = save_path;

        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }
        return item;
    }

    FeigangcaoServerConfig config_;
    std::unique_ptr<Pipeline::FeigangcaoPipeline> pipeline_;
    std::mutex mtx_;
};

FeigangcaoService::FeigangcaoService(const std::string& config_path)
    : m_pHandle(std::make_shared<FeigangcaoServicePrivate>(config_path))
{
}

FeigangcaoService::~FeigangcaoService() = default;

const FeigangcaoServerConfig& FeigangcaoService::config() const
{
    return m_pHandle->config();
}

std::string FeigangcaoService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int FeigangcaoService::runLocalTest(const std::string& image_path,
                                    const std::string& station_id)
{
    return m_pHandle->runLocalTest(image_path, station_id);
}

} // namespace JHDeepCore
