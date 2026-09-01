#include "file_utils.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <ctime>
#endif

#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

JHDeepCore::ServerConfig FileHelper::loadConfig(const std::string& config_path)
{
    JHDeepCore::ServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<std::string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.label_detect_model = node["models"]["label_detect_model"].as<std::string>("");
        cfg.char_detect_model = node["models"]["char_detect_model"].as<std::string>("");
        cfg.ocr_rec_model = node["models"]["ocr_rec_model"].as<std::string>("");
        cfg.ocr_rec_label = node["models"]["ocr_rec_label"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::TiebiaoServerConfig FileHelper::loadTiebiaoConfig(const std::string& config_path)
{
    JHDeepCore::TiebiaoServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<std::string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.label_seg_model = node["models"]["label_seg_model"].as<std::string>("");
        cfg.char_seg_model = node["models"]["char_seg_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
        cfg.direction_cls_model = node["models"]["direction_cls_model"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::DispatchServerConfig FileHelper::loadDispatchConfig(const std::string& config_path)
{
    JHDeepCore::DispatchServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["models"]) {
        cfg.dispatch_classifier_model = node["models"]["dispatch_classifier_model"].as<std::string>("");
        cfg.dispatch_classifier_label = node["models"]["dispatch_classifier_label"].as<std::string>("");
    }

    if (node["dispatch"]) {
        cfg.confidence_threshold = node["dispatch"]["confidence_threshold"].as<float>(cfg.confidence_threshold);
        cfg.default_branch = node["dispatch"]["default_branch"].as<std::string>(cfg.default_branch);
        cfg.dabang_config = node["dispatch"]["dabang_config"].as<std::string>("");
        cfg.tiebiao_config = node["dispatch"]["tiebiao_config"].as<std::string>("");
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::ZbhcServerConfig FileHelper::loadZbhcConfig(const std::string& config_path)
{
    JHDeepCore::ZbhcServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<std::string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
        cfg.char_crop_dir = node["output"]["char_crop_dir"].as<std::string>(cfg.char_crop_dir);
    }

    if (node["models"]) {
        cfg.det1_model = node["models"]["det1_model"].as<std::string>("");
        cfg.det2_model = node["models"]["det2_model"].as<std::string>("");
        cfg.seg_model = node["models"]["seg_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
        cfg.direction_cls_model = node["models"]["direction_cls_model"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::LuqianServerConfig FileHelper::loadLuqianConfig(const std::string& config_path)
{
    JHDeepCore::LuqianServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<std::string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
        cfg.char_crop_dir = node["output"]["char_crop_dir"].as<std::string>(cfg.char_crop_dir);
    }

    if (node["models"]) {
        cfg.det_model = node["models"]["det_model"].as<std::string>("");
        cfg.seg_model = node["models"]["seg_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
        cfg.ocr_model2 = node["models"]["ocr_model2"].as<std::string>("");
        cfg.ocr_label2 = node["models"]["ocr_label2"].as<std::string>("");
        cfg.direction_cls_model = node["models"]["direction_cls_model"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::XintiangangServerConfig FileHelper::loadXintiangangConfig(const std::string& config_path)
{
    JHDeepCore::XintiangangServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.det_model = node["models"]["det_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
        cfg.direction_cls_model = node["models"]["direction_cls_model"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::HuaxinServerConfig FileHelper::loadHuaxinConfig(const std::string& config_path)
{
    JHDeepCore::HuaxinServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
        cfg.url_path = node["server"]["url_path"].as<std::string>(cfg.url_path);
    }

    if (node["output"]) {
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.det1_model = node["models"]["det1_model"].as<std::string>("");
        cfg.det2_model = node["models"]["det2_model"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::GuokuacheServerConfig FileHelper::loadGuokuacheConfig(const std::string& config_path)
{
    JHDeepCore::GuokuacheServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.det1_model = node["models"]["det1_model"].as<std::string>("");
        cfg.det2_model = node["models"]["det2_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::FujianServerConfig FileHelper::loadFujianConfig(const std::string& config_path)
{
    JHDeepCore::FujianServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
        cfg.roi_crop_dir = node["output"]["roi_crop_dir"].as<std::string>(cfg.roi_crop_dir);
        cfg.char_crop_dir = node["output"]["char_crop_dir"].as<std::string>(cfg.char_crop_dir);
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    // 解析 [[x1,y1],[x2,y2]] 两点正矩形 -> x,y,w,h
    auto parseRoi = [](const YAML::Node& n, std::array<int, 4>& out) {
        if (!n || !n.IsSequence() || n.size() != 2) return;
        YAML::Node p1 = n[0];
        YAML::Node p2 = n[1];
        if (!p1.IsSequence() || p1.size() != 2 || !p2.IsSequence() || p2.size() != 2) return;
        int x1 = p1[0].as<int>(), y1 = p1[1].as<int>();
        int x2 = p2[0].as<int>(), y2 = p2[1].as<int>();
        out[0] = std::min(x1, x2);
        out[1] = std::min(y1, y2);
        out[2] = std::abs(x2 - x1);
        out[3] = std::abs(y2 - y1);
    };

    if (node["stations"] && node["stations"].IsSequence()) {
        for (const auto& st : node["stations"]) {
            JHDeepCore::FujianStationConfig sc;
            sc.station_id = st["station_id"].as<int>(0);
            sc.det_model_path = st["det_model_path"].as<std::string>("");
            sc.rec_model_path = st["rec_model_path"].as<std::string>("");
            sc.rec_label_path = st["rec_label_path"].as<std::string>("");
            // roi1、roi2… 递增键名连续解析，直到缺失为止（数量不限）
            for (int ri = 1; ; ++ri) {
                std::string key = "roi" + std::to_string(ri);
                if (!st[key]) break;
                std::array<int, 4> roi = {0, 0, 0, 0};
                parseRoi(st[key], roi);
                sc.rois.push_back(roi);
            }
            cfg.stations.push_back(sc);
        }
    }

    return cfg;
}

JHDeepCore::TedaiJuanquServerConfig FileHelper::loadTedaiJuanquConfig(const std::string& config_path)
{
    JHDeepCore::TedaiJuanquServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
        cfg.url_path = node["server"]["url_path"].as<std::string>(cfg.url_path);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
        cfg.liuzhi_roi_file = node["inference"]["liuzhi_roi_file"].as<std::string>(cfg.liuzhi_roi_file);
        cfg.classify_model = node["inference"]["classify_model"].as<std::string>("");
    }

    if (node["cameras"] && node["cameras"].IsSequence()) {
        for (const auto& cam : node["cameras"]) {
            JHDeepCore::TedaiJuanquCameraConfig cc;
            cc.camera_id = cam["camera_id"].as<int>(0);
            cc.det_model = cam["det_model"].as<std::string>("");
            cfg.cameras.push_back(cc);
        }
    }

    return cfg;
}

JHDeepCore::ZbsltjServerConfig FileHelper::loadZbsltjConfig(const std::string& config_path)
{
    JHDeepCore::ZbsltjServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.billet_det_model = node["models"]["billet_det_model"].as<std::string>("");
        cfg.char_seg_model = node["models"]["char_seg_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

JHDeepCore::GxJingzhengServerConfig FileHelper::loadGxJingzhengConfig(const std::string& config_path)
{
    JHDeepCore::GxJingzhengServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.service_name = node["server"]["service_name"].as<std::string>(cfg.service_name);
        cfg.host = node["server"]["host"].as<std::string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<std::string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<std::string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<std::string>(cfg.log_dir);
        cfg.char_crop_dir = node["output"]["char_crop_dir"].as<std::string>(cfg.char_crop_dir);
    }

    if (node["models"]) {
        cfg.dingwei_model = node["models"]["dingwei_model"].as<std::string>("");
        cfg.dingwei_label = node["models"]["dingwei_label"].as<std::string>("");
        cfg.seg_model = node["models"]["seg_model"].as<std::string>("");
        cfg.seg_label = node["models"]["seg_label"].as<std::string>("");
        cfg.direction_cls_model = node["models"]["direction_cls_model"].as<std::string>("");
        cfg.ocr_model = node["models"]["ocr_model"].as<std::string>("");
        cfg.ocr_label = node["models"]["ocr_label"].as<std::string>("");
        cfg.tiebiao_config = node["models"]["tiebiao_config"].as<std::string>("");
    }

    if (node["classes"]) {
        cfg.zifu_class_name = node["classes"]["zifu_class_name"].as<std::string>(cfg.zifu_class_name);
        cfg.gangbiao_class_name = node["classes"]["gangbiao_class_name"].as<std::string>(cfg.gangbiao_class_name);
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<std::string>("cuda");
    }

    return cfg;
}

std::vector<std::string> FileHelper::splitStringByCsharp(const std::string& str)
{
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, '#')) {
        tokens.push_back(token);
    }
    return tokens;
}

bool FileHelper::ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    if (_access(path.c_str(), 0) != 0) {
        if (_mkdir(path.c_str()) != 0) {
            std::string parent = path;
            size_t pos = parent.find_last_of("\\/");
            if (pos != std::string::npos) {
                ensureDirectoryExists(parent.substr(0, pos));
            }
            _mkdir(path.c_str());
        }
    }
#else
    if (access(path.c_str(), F_OK) != 0) {
        if (mkdir(path.c_str(), 0755) != 0) {
            std::string parent = path;
            size_t pos = parent.find_last_of("\\/");
            if (pos != std::string::npos) {
                ensureDirectoryExists(parent.substr(0, pos));
            }
            mkdir(path.c_str(), 0755);
        }
    }
#endif
    return true;
}

void FileHelper::writeLog(std::ofstream& fout, const std::string& msg)
{
#ifdef _WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    fout << msg << std::endl;
    fout << "timestamp[" << sys.wYear << std::setfill('0') << std::setw(2) << sys.wMonth
         << std::setfill('0') << std::setw(2) << sys.wDay
         << std::setfill('0') << std::setw(2) << sys.wHour
         << std::setfill('0') << std::setw(2) << sys.wMinute
         << std::setfill('0') << std::setw(2) << sys.wSecond << "]" << std::endl;
#else
    time_t now = time(nullptr);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", localtime(&now));
    fout << msg << std::endl;
    fout << "timestamp[" << timestamp << "]" << std::endl;
#endif
}

void FileHelper::createSplitDirectories(const std::string& split_dir, int station_id, const tm* t)
{
    std::string base = cv::format("%s\\station_%02d\\%d%02d%02d",
        split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    ensureDirectoryExists(base);

    const char* subdirs[] = {"biaoqian\\ok", "biaoqian\\ng",
                             "gangyin\\ok", "gangyin\\ng",
                             "penma\\ok", "penma\\ng"};
    for (auto s : subdirs) {
        ensureDirectoryExists(base + "\\" + s);
    }
}
