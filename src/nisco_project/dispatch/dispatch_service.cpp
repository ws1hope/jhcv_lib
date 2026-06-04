#include "JHDeepCore.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class DispatchServicePrivate {
public:
    explicit DispatchServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadDispatchConfig(config_path))
    {
        int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

        classifier_ = std::make_unique<Classifier>(
            config_.dispatch_classifier_model,
            config_.dispatch_classifier_label, dev_id);
        std::cout << "[OK] Dispatch classifier loaded: "
                  << config_.dispatch_classifier_model << std::endl;

        dabang_service_ = std::make_unique<OCRService>(config_.dabang_config);
        std::cout << "[OK] DabangJiguang service loaded: "
                  << config_.dabang_config << std::endl;

        tiebiao_service_ = std::make_unique<TiebiaoService>(config_.tiebiao_config);
        std::cout << "[OK] Tiebiao service loaded: "
                  << config_.tiebiao_config << std::endl;

        warmup();
    }

    const DispatchServerConfig& config() const { return config_; }

    std::string dispatch(const cv::Mat& image)
    {
        std::vector<cv::Mat> images = {image};
        std::vector<ClassificationResult> results;
        classifier_->process(images, results);

        if (results.empty()) {
            std::cerr << "[WARN] Dispatch classifier returned no result, using default: "
                      << config_.default_branch << std::endl;
            return config_.default_branch;
        }

        if (results[0].confidence < config_.confidence_threshold) {
            std::cout << "[INFO] Low confidence (" << results[0].confidence
                      << " < " << config_.confidence_threshold
                      << "), using default: " << config_.default_branch << std::endl;
            return config_.default_branch;
        }

        std::string branch = results[0].class_name;
        std::cout << "[INFO] Dispatch -> " << branch
                  << " (confidence: " << results[0].confidence << ")" << std::endl;
        return branch;
    }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Dispatch.txt";
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

        std::vector<std::string> picture_path_array =
            FileHelper::splitStringByCsharp(picture_path);

        std::string branch = config_.default_branch;
        for (const auto& path : picture_path_array) {
            cv::Mat img = cv::imread(path);
            if (!img.data) continue;
            branch = dispatch(img);
            break;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path
             << " dispatch:" << branch << std::endl;

        std::string result;
        if (branch == "tiebiao") {
            result = tiebiao_service_->handleRequest(req_body);
        } else {
            result = dabang_service_->handleRequest(req_body);
        }

        json result_json = json::parse(result);
        result_json["dispatch_branch"] = branch;
        result = result_json.dump();

        fout.close();
        return result;
    }

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     int station_id)
    {
        std::cout << "=== Dispatch Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << std::endl;

        std::vector<std::string> picture_path_array =
            FileHelper::splitStringByCsharp(image_path);

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat first_img;
        for (const auto& path : picture_path_array) {
            first_img = cv::imread(path);
            if (first_img.data) break;
        }

        std::string branch = config_.default_branch;
        if (first_img.data) {
            branch = dispatch(first_img);
        }

        std::cout << std::endl;
        std::cout << ">>> Dispatched to: " << branch << std::endl;
        std::cout << std::endl;

        std::string result;
        if (branch == "0") {
            result = tiebiao_service_->handleRequest(
                buildTestJson(heat_number, station_id, image_path));
        } else {
            result = dabang_service_->handleRequest(
                buildTestJson(heat_number, station_id, image_path));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        json result_json = json::parse(result);
        result_json["dispatch_branch"] = branch;

        std::cout << "=== Result ===" << std::endl;
        std::cout << result_json.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    DispatchServerConfig config_;
    std::unique_ptr<Classifier> classifier_;
    std::unique_ptr<OCRService> dabang_service_;
    std::unique_ptr<TiebiaoService> tiebiao_service_;
    std::mutex mtx_;

private:
    void warmup()
    {
        std::cout << "[INFO] Warming up dispatch classifier..." << std::endl;
        std::vector<cv::Mat> dummy_imgs = {
            cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};
        std::vector<ClassificationResult> results;
        classifier_->process(dummy_imgs, results);
        std::cout << "[OK] Dispatch classifier warmed up." << std::endl;
    }

    static std::string buildTestJson(const std::string& heat_number,
                                     int station_id,
                                     const std::string& image_path)
    {
        json req;
        req["heat_number"] = heat_number;
        req["station_id"] = station_id;
        req["picture_path"] = image_path;
        return req.dump();
    }
};

DispatchService::DispatchService(const std::string& config_path)
    : m_pHandle(std::make_shared<DispatchServicePrivate>(config_path))
{
}

DispatchService::~DispatchService() = default;

const DispatchServerConfig& DispatchService::config() const
{
    return m_pHandle->config();
}

std::string DispatchService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int DispatchService::runLocalTest(const std::string& image_path,
                                  const std::string& heat_number,
                                  int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
