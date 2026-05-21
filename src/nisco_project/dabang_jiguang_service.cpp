#include "JHDeepCore.h"
#include "file_utils.h"
#include "infer_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class OCRServicePrivate {
public:
    explicit OCRServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadConfig(config_path))
    {
        int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

        det_label_ = std::make_unique<Detector>(
            config_.label_detect_model, "", dev_id);
        std::cout << "[OK] Label detect model loaded: " << config_.label_detect_model << std::endl;

        det_char_ = std::make_unique<Detector>(
            config_.char_detect_model, "", dev_id);
        std::cout << "[OK] Char detect model loaded: " << config_.char_detect_model << std::endl;

        ocr_ = std::make_unique<OCRRecognizer>(
            config_.ocr_rec_model, config_.ocr_rec_label, dev_id);
        std::cout << "[OK] OCR model loaded: " << config_.ocr_rec_model << std::endl;

        warmup();
    }

    ~OCRServicePrivate()
    {
        det_label_.reset();
        det_char_.reset();
        ocr_.reset();
    }

    const ServerConfig& config() const { return config_; }

    std::vector<DabangJiguangResult> recognize(
        const std::vector<std::string>& picture_path_array,
        int station_id,
        const std::string& heat_number,
        bool verbose,
        std::ofstream* pfout = nullptr)
    {
        std::vector<DabangJiguangResult> results;

        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++)
        {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            if (!src_img.data) {
                std::cerr << "empty image: " << picture_path_array[pic_number] << std::endl;
                DabangJiguangResult item;
                item.picture_id = pic_number + 1;
                item.state_flag = "NG";
                results.push_back(item);

                if (pfout) {
                    FileHelper::writeLog(*pfout, "detect failed! empty image");
                    *pfout << "****************************************" << std::endl;
                }
                continue;
            }

            results.push_back(recognizeSingle(src_img, station_id, heat_number,
                                              pic_number, verbose));
        }

        return results;
    }

    DabangJiguangResult recognizeSingle(
        const cv::Mat& src_img,
        int station_id,
        const std::string& heat_number,
        int pic_number,
        bool verbose)
    {
        auto infer_start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Mat> label_imgs = {src_img};
        std::vector<DetectionResult> label_results;
        det_label_->process(label_imgs, label_results);
        DetectionResult label_result = label_results.empty() ? DetectionResult{} : label_results[0];

        if (verbose) {
            std::cout << "[DEBUG] pic " << pic_number + 1
                 << " labels: " << label_result.num_detections << std::endl;
        }

        std::string ocr_combined;

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

                std::vector<cv::Mat> char_imgs = {roi_img};
                std::vector<DetectionResult> char_results;
                det_char_->process(char_imgs, char_results);
                DetectionResult char_result = char_results.empty() ? DetectionResult{} : char_results[0];

                if (verbose) {
                    std::cout << "  chars: " << char_result.num_detections << std::endl;
                }

                std::vector<Detection> valid_char_dets;
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

                    std::vector<cv::Mat> ocr_imgs = {char_img_bgr};
                    std::vector<OCRResult> ocr_results;
                    ocr_->process(ocr_imgs, ocr_results);
                    OCRResult ocr_result = ocr_results.empty() ? OCRResult{} : ocr_results[0];

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

        DabangJiguangResult result;
        result.picture_id = pic_number + 1;
        result.zifu_type = "Penma";

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

            result.state_flag = "OK";
            result.ocr_text = ocr_combined;
            result.picture_path = save_picture_name;

            std::string dd = std::to_string(station_id) + ";OK;Penma;" + ocr_combined + ";" + save_picture_name;
            if (verbose) {
                std::cout << ">>> OK " << dd << std::endl;
            }
        } else {
            cv::imwrite(save_picture_name, src_img);
            cv::imwrite(save_picture_name_split_ng, src_img);

            result.state_flag = "NG";
            result.picture_path = save_picture_name;

            std::string dd = std::to_string(station_id) + ";NG;Penma;" + save_picture_name;
            if (verbose) {
                std::cout << ">>> NG " << dd << std::endl;
            }
        }

        return result;
    }

    std::string handleRequest(const std::string& req_body)
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

        std::vector<DabangJiguangResult> results = recognize(
            picture_path_array, station_id, heat_number, false, &fout);

        json root_all;
        root_all["station_id"] = station_id;
        json array_result = json::array();
        for (const auto& r : results) {
            json item;
            item["picture_id"] = r.picture_id;
            item["state_flag"] = r.state_flag;
            item["zifu_type"] = r.zifu_type;
            item["result"] = r.ocr_text;
            item["picture_path"] = r.picture_path;
            array_result.push_back(item);
        }
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
        std::vector<DabangJiguangResult> results = recognize(
            picture_path_array, station_id, heat_number, true);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        json root_all;
        root_all["station_id"] = station_id;
        json array_result = json::array();
        for (const auto& r : results) {
            json item;
            item["picture_id"] = r.picture_id;
            item["state_flag"] = r.state_flag;
            item["zifu_type"] = r.zifu_type;
            item["result"] = r.ocr_text;
            item["picture_path"] = r.picture_path;
            array_result.push_back(item);
        }
        root_all["all_results"] = array_result;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root_all.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    ServerConfig config_;
    std::unique_ptr<Detector> det_label_;
    std::unique_ptr<Detector> det_char_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::mutex mtx_;

    void warmup()
    {
        std::cout << "[INFO] Warming up models..." << std::endl;
        std::vector<cv::Mat> dummy_imgs = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

        std::vector<DetectionResult> label_results;
        det_label_->process(dummy_imgs, label_results);
        int label_dets = label_results.empty() ? 0 : label_results[0].num_detections;
        std::cout << "[OK] Label detector warmed up (dummy detections: "
             << label_dets << ")" << std::endl;

        std::vector<DetectionResult> char_results;
        det_char_->process(dummy_imgs, char_results);
        int char_dets = char_results.empty() ? 0 : char_results[0].num_detections;
        std::cout << "[OK] Char detector warmed up (dummy detections: "
             << char_dets << ")" << std::endl;

        std::vector<OCRResult> ocr_results;
        ocr_->process(dummy_imgs, ocr_results);
        std::cout << "[OK] OCR recognizer warmed up" << std::endl;

        std::cout << "[OK] Warmup complete." << std::endl;
    }
};

OCRService::OCRService(const std::string& config_path)
    : m_pHandle(std::make_shared<OCRServicePrivate>(config_path))
{
}

OCRService::~OCRService() = default;

const ServerConfig& OCRService::config() const
{
    return m_pHandle->config();
}

std::string OCRService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int OCRService::runLocalTest(const std::string& image_path,
                             const std::string& heat_number,
                             int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
