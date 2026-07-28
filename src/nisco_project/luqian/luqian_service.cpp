#include "JHDeepCore.h"
#include "luqian_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

// 单图置信度 = 所有结果(目标, ocr_text 非空)置信度的最小值；无结果返回 0
static float computeImageConfidence(const Pipeline::LuqianPipelineResult& r)
{
    float img_min = 1.0f;
    bool any = false;
    for (auto& target : r.targets) {
        if (target.ocr_text.empty()) continue;
        img_min = std::min(img_min, target.ocr_confidence);
        any = true;
    }
    return any ? img_min : 0.0f;
}

// 多图择优：只保留图置信度最大的那张图（confs 非空且与 array_result 等长时调用）
static size_t keepBestImageConfidence(json& array_result,
                                      const std::vector<float>& confs)
{
    size_t best = 0;
    for (size_t i = 1; i < confs.size(); i++) {
        if (confs[i] > confs[best]) best = i;
    }
    json best_item = array_result[best];
    array_result = json::array();
    array_result.push_back(std::move(best_item));
    return best;
}

class LuqianServicePrivate {
public:
    explicit LuqianServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadLuqianConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::LuqianPipeline>(config_);
    }

    const LuqianServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Luqian.txt";
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

        json array_result = json::array();
        std::vector<float> img_confs;
        std::vector<bool> has_detections;
        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;

            if (!src_img.data) {
                item["picture_id"] = pic_number + 1;
                item["state_flag"] = "NG";
                item["result"] = "";
                item["zifu_type"] = "Penma";
                item["penma_version"] = "None";
                item["picture_path"] = "";
                array_result.push_back(item);
                img_confs.push_back(0.0f);
                has_detections.push_back(false);
                fout << "detect failed! empty image" << std::endl;
                continue;
            }

            Pipeline::LuqianPipelineResult pipeline_result = pipeline_->process(src_img, false, heat_number);
            img_confs.push_back(computeImageConfidence(pipeline_result));
            has_detections.push_back(!pipeline_result.det_detections.empty());

            // 各模型分段耗时写进日志（每张图一行）。run 在 cuda 下含 H2D 拷贝，ten 预期≈0。
            {
                const auto& T = pipeline_result.timing;
                auto put = [&](const char* name, const InferenceTiming& t) {
                    fout << " " << name << "(n=" << t.count
                         << " prep=" << t.preprocess_ms;
                    if (t.h2d_split) {
                        fout << " h2d=" << t.h2d_ms
                             << " d2h=" << t.d2h_ms
                             << " infer=" << (t.run_ms - t.h2d_ms - t.d2h_ms);
                    } else {
                        fout << " ten=" << t.tensor_ms
                             << " run=" << t.run_ms;
                    }
                    fout << ")";
                };
                fout << "[timing] pic=" << (pic_number + 1) << " device=" << T.device;
                put("det", T.det);
                put("seg", T.seg);
                put("cls", T.cls);
                put("ocr", T.ocr);
                put("ocr2", T.ocr2);
                fout << " total=" << T.total_ms << "ms" << std::endl;
            }

            // 每个目标的两次识别(ocr1/ocr2)与最终结果(final)写入日志（ocr1/ocr2 不再绘于结果图）
            for (int ti = 0; ti < (int)pipeline_result.targets.size(); ti++) {
                const auto& t = pipeline_result.targets[ti];
                fout << "[ocr] pic=" << (pic_number + 1) << " target=" << (ti + 1)
                     << " ocr1=\"" << t.ocr1_text << "\""
                     << " ocr2=\"" << t.ocr2_text << "\""
                     << " final=\"" << t.ocr_text << "\"" << std::endl;
            }

            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);

            std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            FileHelper::ensureDirectoryExists(folderPath);

            std::string save_picture_name = cv::format(
                "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_%d.jpg",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec, pic_number);

            if (!pipeline_result.annotated_image.empty()) {
                cv::imwrite(save_picture_name, pipeline_result.annotated_image);
            } else if (!src_img.empty()) {
                cv::imwrite(save_picture_name, src_img);
            }

            saveCharCrops(pipeline_result, pic_number, t);

            std::string combined_result;
            for (auto& target : pipeline_result.targets) {
                if (!target.ocr_text.empty()) {
                    if (!combined_result.empty()) combined_result += "#";
                    combined_result += target.ocr_text;
                }
            }

            item["picture_id"] = pic_number + 1;
            item["state_flag"] = combined_result.empty() ? "NG" : "OK";
            item["result"] = combined_result;
            item["zifu_type"] = "Penma";
            item["penma_version"] = combined_result.empty() ? "None" : "new";
            item["picture_path"] = save_picture_name;

            array_result.push_back(item);
        }

        size_t best_idx = 0;
        if (!img_confs.empty()) {
            best_idx = keepBestImageConfidence(array_result, img_confs);
            std::cout << "[BEST] idx=" << best_idx
                      << " conf=" << img_confs[best_idx]
                      << " kept=" << array_result.size() << "/" << img_confs.size() << std::endl;
        }

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;
        root_all["duanmian"] = (!has_detections.empty() && has_detections[best_idx]) ? "yes" : "no";

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
        json array_result = json::array();
        std::vector<float> img_confs;
        std::vector<bool> has_detections;

        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;

            if (!src_img.data) {
                std::cerr << "[ERROR] Cannot read image: " << picture_path_array[pic_number] << std::endl;
                item["picture_id"] = pic_number + 1;
                item["state_flag"] = "NG";
                item["result"] = "";
                item["zifu_type"] = "Penma";
                item["penma_version"] = "None";
                item["picture_path"] = "";
                array_result.push_back(item);
                img_confs.push_back(0.0f);
                has_detections.push_back(false);
                continue;
            }

            Pipeline::LuqianPipelineResult pipeline_result = pipeline_->process(src_img, true, heat_number);
            img_confs.push_back(computeImageConfidence(pipeline_result));
            has_detections.push_back(!pipeline_result.det_detections.empty());

            std::string combined_result;
            for (auto& target : pipeline_result.targets) {
                if (!target.ocr_text.empty()) {
                    if (!combined_result.empty()) combined_result += "#";
                    combined_result += target.ocr_text;
                }
            }

            item["picture_id"] = pic_number + 1;
            item["state_flag"] = combined_result.empty() ? "NG" : "OK";
            item["result"] = combined_result;
            item["zifu_type"] = "Penma";
            item["penma_version"] = combined_result.empty() ? "None" : "new";
            item["picture_path"] = "";

            array_result.push_back(item);

            if (!pipeline_result.annotated_image.empty()) {
                std::string save_path = "luqian_result_" + std::to_string(pic_number) + ".jpg";
                cv::imwrite(save_path, pipeline_result.annotated_image);
                std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
            }

            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);
            saveCharCrops(pipeline_result, pic_number, t);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        size_t best_idx = 0;
        if (!img_confs.empty()) {
            best_idx = keepBestImageConfidence(array_result, img_confs);
            std::cout << "[BEST] idx=" << best_idx
                      << " conf=" << img_confs[best_idx]
                      << " kept=" << array_result.size() << "/" << img_confs.size() << std::endl;
        }

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;
        root_all["duanmian"] = (!has_detections.empty() && has_detections[best_idx]) ? "yes" : "no";

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root_all.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    // 保存每个字符的方向矫正后裁剪图（OCR 输入）到 char_crop_dir，扁平命名：
    // <char_crop_dir>\YYYYMMDDHHMMSS_p<pic>_t<target>_c<char>.jpg
    // char_crop_dir 为空时跳过
    void saveCharCrops(const Pipeline::LuqianPipelineResult& result,
                       int pic_number, const tm* t)
    {
        if (config_.char_crop_dir.empty()) return;
        FileHelper::ensureDirectoryExists(config_.char_crop_dir);
        for (int ti = 0; ti < (int)result.targets.size(); ti++) {
            const auto& target = result.targets[ti];
            for (int ci = 0; ci < (int)target.chars.size(); ci++) {
                const cv::Mat& crop = target.chars[ci].image_after_flip;
                if (crop.empty()) continue;
                std::string save_name = cv::format("%s\\%d%02d%02d%02d%02d%02d_p%d_t%d_c%d.jpg",
                    config_.char_crop_dir.c_str(),
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                    t->tm_hour, t->tm_min, t->tm_sec,
                    pic_number, ti + 1, ci + 1);
                cv::imwrite(save_name, crop);
            }
        }
    }

    LuqianServerConfig config_;
    std::unique_ptr<Pipeline::LuqianPipeline> pipeline_;
    std::mutex mtx_;
};

LuqianService::LuqianService(const std::string& config_path)
    : m_pHandle(std::make_shared<LuqianServicePrivate>(config_path))
{
}

LuqianService::~LuqianService() = default;

const LuqianServerConfig& LuqianService::config() const
{
    return m_pHandle->config();
}

std::string LuqianService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int LuqianService::runLocalTest(const std::string& image_path,
                                 const std::string& heat_number,
                                 int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
