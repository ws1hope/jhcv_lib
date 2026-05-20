#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <memory>

#include <opencv2/core.hpp>

#include "JHDeepCore.h"
#include "json.hpp"

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string result_dir = "D:\\CharacterDetect\\result";
    std::string split_dir = "D:\\CharacterDetect\\result_split";
    std::string log_dir = "visual_logs";
    std::string label_detect_model;
    std::string char_detect_model;
    std::string ocr_rec_model;
    std::string ocr_rec_label;
    std::string device = "cuda";
};

class FileHelper {
public:
    FileHelper() = delete;

    static ServerConfig loadConfig(const std::string& config_path);

    static std::vector<std::string> splitStringByCsharp(const std::string& str);

    static bool ensureDirectoryExists(const std::string& path);

    static void writeLog(std::ofstream& fout, const std::string& msg);

    static void createSplitDirectories(const std::string& split_dir, int station_id, const tm* t);
};

class InferHelper {
public:
    InferHelper() = delete;

    static cv::Rect safeROI(int x, int y, int w, int h, int img_w, int img_h);

    static std::string sortCharsByPosition(
        const std::vector<JHDeepCore::Detection>& char_dets,
        const std::vector<std::string>& char_texts);
};

class OCRService {
public:
    explicit OCRService(const std::string& config_path);

    ~OCRService();

    OCRService(const OCRService&) = delete;
    OCRService& operator=(const OCRService&) = delete;

    const ServerConfig& config() const;

    nlohmann::json recognize(const std::vector<std::string>& picture_path_array,
                             int station_id,
                             const std::string& heat_number,
                             bool verbose,
                             std::ofstream* pfout = nullptr);

    nlohmann::json handleRequest(const std::string& req_body);

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     int station_id);

private:
    ServerConfig config_;
    std::unique_ptr<JHDeepCore::Detector> det_label_;
    std::unique_ptr<JHDeepCore::Detector> det_char_;
    std::unique_ptr<JHDeepCore::OCRRecognizer> ocr_;
    std::mutex mtx_;

    void warmup_();

    nlohmann::json recognizeSingle_(const cv::Mat& src_img,
                                    int station_id,
                                    const std::string& heat_number,
                                    int pic_number,
                                    bool verbose);
};
