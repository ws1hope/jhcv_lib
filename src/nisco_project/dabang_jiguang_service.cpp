#include "dabang_jiguang_service.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <direct.h>
#include <io.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include "httplib.h"
#include "JHDeepCore.h"

using json = nlohmann::json;

ServerConfig FileHelper::loadConfig(const std::string& config_path)
{
    ServerConfig cfg;
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
    return true;
}

void FileHelper::writeLog(std::ofstream& fout, const std::string& msg)
{
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    fout << msg << std::endl;
    fout << "timestamp[" << sys.wYear << std::setfill('0') << std::setw(2) << sys.wMonth
         << std::setfill('0') << std::setw(2) << sys.wDay
         << std::setfill('0') << std::setw(2) << sys.wHour
         << std::setfill('0') << std::setw(2) << sys.wMinute
         << std::setfill('0') << std::setw(2) << sys.wSecond << "]" << std::endl;
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

cv::Rect InferHelper::safeROI(int x, int y, int w, int h, int img_w, int img_h)
{
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, img_w - x);
    h = std::min(h, img_h - y);
    if (w <= 0 || h <= 0) return cv::Rect(0, 0, 0, 0);
    return cv::Rect(x, y, w, h);
}

std::string InferHelper::sortCharsByPosition(
    const std::vector<JHDeepCore::Detection>& char_dets,
    const std::vector<std::string>& char_texts)
{
    if (char_dets.empty()) return "";

    std::vector<std::pair<int, std::string>> chars_with_pos;
    for (int i = 0; i < (int)char_dets.size(); i++) {
        chars_with_pos.emplace_back(char_dets[i].bbox.y, char_texts[i]);
    }

    std::sort(chars_with_pos.begin(), chars_with_pos.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });

    std::string combined;
    bool first = true;
    for (auto& cp : chars_with_pos) {
        combined += cp.second;
        if (first) {
            combined += "#";
            first = false;
        }
    }
    return combined;
}

OCRService::OCRService(const std::string& config_path)
    : config_(FileHelper::loadConfig(config_path))
{
    det_label_ = std::make_unique<JHDeepCore::Detector>(
        config_.label_detect_model, config_.device);
    std::cout << "[OK] Label detect model loaded: " << config_.label_detect_model << std::endl;

    det_char_ = std::make_unique<JHDeepCore::Detector>(
        config_.char_detect_model, config_.device);
    std::cout << "[OK] Char detect model loaded: " << config_.char_detect_model << std::endl;

    JHDeepCore::OCRRecognizer::Params ocr_params;
    ocr_params.rec_model_path = config_.ocr_rec_model;
    ocr_params.rec_label_path = config_.ocr_rec_label;
    ocr_params.device = config_.device;
    ocr_ = std::make_unique<JHDeepCore::OCRRecognizer>(ocr_params);
    std::cout << "[OK] OCR model loaded: " << config_.ocr_rec_model << std::endl;

    warmup_();
}

OCRService::~OCRService()
{
    det_label_.reset();
    det_char_.reset();
    ocr_.reset();
}

const ServerConfig& OCRService::config() const { return config_; }

json OCRService::recognize(const std::vector<std::string>& picture_path_array,
                           int station_id,
                           const std::string& heat_number,
                           bool verbose,
                           std::ofstream* pfout)
{
    json root_all;
    root_all["station_id"] = station_id;
    json array_result = json::array();

    for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++)
    {
        cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

        if (!src_img.data) {
            std::cerr << "empty image: " << picture_path_array[pic_number] << std::endl;
            json item;
            item["picture_id"] = pic_number + 1;
            item["state_flag"] = "NG";
            item["zifu_type"] = "";
            item["result"] = "";
            item["picture_path"] = "";
            array_result.push_back(item);

            if (pfout) {
                FileHelper::writeLog(*pfout, "detect failed! empty image");
                *pfout << "****************************************" << std::endl;
            }
            continue;
        }

        json item = recognizeSingle_(src_img, station_id, heat_number,
                                     pic_number, verbose);
        array_result.push_back(item);
    }

    root_all["all_results"] = array_result;
    return root_all;
}

json OCRService::handleRequest(const std::string& req_body)
{
    std::lock_guard<std::mutex> lock(mtx_);

    time_t currentTime = time(NULL);
    char chCurrentTime[256];
    strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
    std::string stCurrentTime = chCurrentTime;
    std::string filename = stCurrentTime + "@OCR.txt";
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

    json root_all = recognize(picture_path_array, station_id, heat_number,
                              false, &fout);

    std::cout << "[RESULT] " << root_all.dump() << std::endl;
    fout.close();

    return root_all;
}

int OCRService::runLocalTest(const std::string& image_path,
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
    json result = recognize(picture_path_array, station_id, heat_number, true);
    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << std::endl;
    std::cout << "=== Result ===" << std::endl;
    std::cout << result.dump(2) << std::endl;
    std::cout << "total time: " << total_ms << " ms" << std::endl;

    return 0;
}

void OCRService::warmup_()
{
    std::cout << "[INFO] Warming up models..." << std::endl;
    cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    auto label_result = det_label_->DetectSingle(dummy);
    std::cout << "[OK] Label detector warmed up (dummy detections: "
         << label_result.num_detections << ")" << std::endl;

    auto char_result = det_char_->DetectSingle(dummy);
    std::cout << "[OK] Char detector warmed up (dummy detections: "
         << char_result.num_detections << ")" << std::endl;

    JHDeepCore::OCRResult ocr_result = ocr_->Recognize(dummy);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

json OCRService::recognizeSingle_(const cv::Mat& src_img,
                                  int station_id,
                                  const std::string& heat_number,
                                  int pic_number,
                                  bool verbose)
{
    json item;

    auto infer_start = std::chrono::high_resolution_clock::now();

    JHDeepCore::DetectionResult label_result = det_label_->DetectSingle(src_img);

    if (verbose) {
        std::cout << "[DEBUG] pic " << pic_number + 1
             << " labels: " << label_result.num_detections << std::endl;
    }

    std::string ocr_combined;
    bool success_flag = false;

    if (label_result.num_detections > 0) {
        for (int li = 0; li < label_result.num_detections; li++) {
            auto& label_det = label_result.detections[li];

            if (verbose) {
                std::cout << "  label[" << li << "] class=" << label_det.class_name
                     << " conf=" << label_det.confidence
                     << " bbox=(" << label_det.bbox.x << "," << label_det.bbox.y
                     << "," << label_det.bbox.width << "," << label_det.bbox.height
                     << ")" << std::endl;
            }

            cv::Rect label_roi = InferHelper::safeROI(
                label_det.bbox.x, label_det.bbox.y,
                label_det.bbox.width, label_det.bbox.height,
                src_img.cols, src_img.rows);

            if (label_roi.area() <= 0) continue;

            cv::Mat roi_img = src_img(label_roi).clone();

            JHDeepCore::DetectionResult char_result = det_char_->DetectSingle(roi_img);

            if (verbose) {
                std::cout << "  chars: " << char_result.num_detections << std::endl;
            }

            std::vector<JHDeepCore::Detection> valid_char_dets;
            std::vector<std::string> char_texts;

            for (int ci = 0; ci < char_result.num_detections; ci++) {
                auto& ch = char_result.detections[ci];
                cv::Rect ch_roi = InferHelper::safeROI(
                    ch.bbox.x, ch.bbox.y,
                    ch.bbox.width, ch.bbox.height,
                    roi_img.cols, roi_img.rows);

                if (ch_roi.area() <= 0) continue;

                cv::Mat char_img = roi_img(ch_roi).clone();
                if (char_img.empty()) continue;

                cv::Mat char_img_bgr;
                if (char_img.channels() == 1) {
                    cv::cvtColor(char_img, char_img_bgr, cv::COLOR_GRAY2BGR);
                } else {
                    char_img_bgr = char_img;
                }

                JHDeepCore::OCRResult ocr_result = ocr_->Recognize(char_img_bgr);

                std::string char_text;
                if (!ocr_result.boxes.empty()) {
                    char_text = ocr_result.boxes[0].text;
                }

                if (verbose) {
                    std::cout << "    char[" << ci << "] text=\""
                         << char_text << "\" conf=" << ch.confidence << std::endl;
                }

                valid_char_dets.push_back(ch);
                char_texts.push_back(char_text);
            }

            std::cout << "[DEBUG] valid_char_dets.size()=" << valid_char_dets.size()
                 << ", char_texts.size()=" << char_texts.size() << std::endl;
            for (int i = 0; i < (int)char_texts.size(); i++) {
                std::cout << "[DEBUG]   char[" << i << "] text=\"" << char_texts[i]
                     << "\" bbox=(" << valid_char_dets[i].bbox.x << ","
                     << valid_char_dets[i].bbox.y << ","
                     << valid_char_dets[i].bbox.width << ","
                     << valid_char_dets[i].bbox.height
                     << ") conf=" << valid_char_dets[i].confidence << std::endl;
            }

            std::string line_result = InferHelper::sortCharsByPosition(valid_char_dets, char_texts);

            std::cout << "[DEBUG] line_result=\"" << line_result << "\"" << std::endl;

            if (!line_result.empty()) {
                if (!ocr_combined.empty()) ocr_combined += ",";
                ocr_combined += line_result;
                std::cout << "[DEBUG] ocr_combined=\"" << ocr_combined << "\"" << std::endl;
            }
        }
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) {
        std::cout << "inference time: " << infer_ms << " ms" << std::endl;
    }

    time_t currtime = time(NULL);
    tm* t = localtime(&currtime);

    std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
        config_.result_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    FileHelper::ensureDirectoryExists(folderPath);
    FileHelper::createSplitDirectories(config_.split_dir, station_id, t);

    std::string save_picture_name = cv::format("%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_%d.jpg",
        config_.result_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    std::string save_picture_name_split_ok = cv::format("%s\\station_%02d\\%d%02d%02d\\penma\\ok\\%d%02d%02d%02d%02d%02d_%d.jpg",
        config_.split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    std::string save_picture_name_split_ng = cv::format("%s\\station_%02d\\%d%02d%02d\\penma\\ng\\%d%02d%02d%02d%02d%02d_%d.jpg",
        config_.split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    std::string dd;

    if (!ocr_combined.empty()) {
        bool isOK = false;
        std::string ocr_no_hash;
        for (auto c : ocr_combined) {
            if (c != '#') ocr_no_hash += c;
        }
        if ((int)ocr_no_hash.length() >= 8) {
            std::string first8 = ocr_no_hash.substr(0, 8);
            if (first8 == heat_number) {
                isOK = true;
                if (verbose) std::cout << "luhao are equal." << std::endl;
            }
        }

        if (isOK) {
            cv::imwrite(save_picture_name_split_ok, src_img);
        } else {
            cv::imwrite(save_picture_name_split_ng, src_img);
        }

        cv::imwrite(save_picture_name, src_img);

        dd = std::to_string(station_id) + ";OK;Penma;" + ocr_combined + ";" + save_picture_name;
        success_flag = true;

        item["picture_id"] = pic_number + 1;
        item["state_flag"] = "OK";
        item["zifu_type"] = "Penma";
        item["result"] = ocr_combined;
        item["picture_path"] = save_picture_name;
    } else {
        cv::imwrite(save_picture_name, src_img);
        cv::imwrite(save_picture_name_split_ng, src_img);

        dd = std::to_string(station_id) + ";NG;Penma;" + save_picture_name;
        success_flag = false;

        item["picture_id"] = pic_number + 1;
        item["state_flag"] = "NG";
        item["zifu_type"] = "Penma";
        item["result"] = "";
        item["picture_path"] = save_picture_name;
    }

    if (verbose) {
        std::cout << (success_flag ? ">>> OK" : ">>> NG") << " " << dd << std::endl;
    }

    return item;
}
