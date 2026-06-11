#pragma once

#include <string>
#include <vector>
#include <fstream>

namespace JHDeepCore {

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

struct TiebiaoServerConfig {
    std::string service_name = "tiebiao";
    std::string host = "0.0.0.0";
    int port = 8081;
    std::string result_dir = "D:\\TiebiaoResult";
    std::string split_dir = "D:\\TiebiaoSplit";
    std::string log_dir = "D:\\TiebiaoLog";
    std::string label_seg_model;
    std::string char_seg_model;
    std::string ocr_model;
    std::string ocr_label;
    std::string direction_cls_model;
    std::string device = "cuda";
};

struct TiebiaoConfig {
    std::string label_seg_model;
    std::string char_seg_model;
    std::string ocr_model;
    std::string ocr_label;
    std::string direction_cls_model;
    std::string device;
};

struct DispatchServerConfig {
    std::string service_name = "dispatch";
    std::string host = "0.0.0.0";
    int port = 8082;
    std::string result_dir = "D:\\DispatchResult";
    std::string log_dir = "dispatch_logs";
    std::string dispatch_classifier_model;
    std::string dispatch_classifier_label;
    float confidence_threshold = 0.8f;
    std::string default_branch = "dabang";
    std::string dabang_config;
    std::string tiebiao_config;
    std::string device = "cuda";
};

struct ZbhcServerConfig {
    std::string service_name = "zbhc";
    std::string host = "0.0.0.0";
    int port = 8083;
    std::string result_dir = "D:\\ZbhcResult";
    std::string split_dir = "D:\\ZbhcSplit";
    std::string log_dir = "D:\\ZbhcLog";
    std::string det1_model;
    std::string det2_model;
    std::string seg_model;
    std::string ocr_model;
    std::string ocr_label;
    std::string device = "cuda";
};

} // namespace JHDeepCore

class FileHelper {
public:
    FileHelper() = delete;

    static JHDeepCore::ServerConfig loadConfig(const std::string& config_path);

    static JHDeepCore::TiebiaoServerConfig loadTiebiaoConfig(const std::string& config_path);

    static JHDeepCore::DispatchServerConfig loadDispatchConfig(const std::string& config_path);

    static JHDeepCore::ZbhcServerConfig loadZbhcConfig(const std::string& config_path);

    static std::vector<std::string> splitStringByCsharp(const std::string& str);

    static bool ensureDirectoryExists(const std::string& path);

    static void writeLog(std::ofstream& fout, const std::string& msg);

    static void createSplitDirectories(const std::string& split_dir, int station_id, const tm* t);
};
