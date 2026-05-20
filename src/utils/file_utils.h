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

} // namespace JHDeepCore

class FileHelper {
public:
    FileHelper() = delete;

    static JHDeepCore::ServerConfig loadConfig(const std::string& config_path);

    static std::vector<std::string> splitStringByCsharp(const std::string& str);

    static bool ensureDirectoryExists(const std::string& path);

    static void writeLog(std::ofstream& fout, const std::string& msg);

    static void createSplitDirectories(const std::string& split_dir, int station_id, const tm* t);
};
