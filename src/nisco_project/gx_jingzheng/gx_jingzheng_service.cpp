#include "JHDeepCore.h"
#include "gx_jingzheng_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class GxJingzhengServicePrivate {
public:
    explicit GxJingzhengServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadGxJingzhengConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::GxJingzhengPipeline>(config_);
    }

    const GxJingzhengServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@GxJingzheng.txt";
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
        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);

            json item;
            item["picture_id"] = pic_number + 1;

            if (!src_img.data) {
                item["state_flag"] = "NG";
                item["zifu_type"] = "";
                item["penma_version"] = "new";
                item["result"] = "";
                item["picture_path"] = "";
                item["duanmian"] = "no";
                array_result.push_back(item);
                fout << "detect failed! empty image" << std::endl;
                continue;
            }

            Pipeline::GxJingzhengPipelineResult pres =
                pipeline_->process(src_img, station_id, heat_number, false);

            // 各模型分段耗时写进日志（每张图一行）。run 在 cuda 下含 H2D 拷贝，ten 预期≈0。
            {
                const auto& T = pres.timing;
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
                fout << " total=" << T.total_ms << "ms" << std::endl;
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

            if (!pres.annotated_image.empty()) {
                cv::imwrite(save_picture_name, pres.annotated_image);
            } else if (!src_img.empty()) {
                cv::imwrite(save_picture_name, src_img);
            }

            saveCharCrops(pres, pic_number, t);

            item["state_flag"] = pres.state_flag;
            item["zifu_type"] = pres.zifu_type;
            item["penma_version"] = pres.penma_version;
            item["result"] = pres.ocr_text;
            item["picture_path"] = save_picture_name;
            item["duanmian"] = pres.duanmian;
            array_result.push_back(item);
        }

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;

        std::cout << "[RESULT] " << root_all.dump() << std::endl;
        fout.close();

        return root_all.dump();
    }

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     int station_id)
    {
        std::cout << "=== Local Test Mode (gx_jingzheng) ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << std::endl;

        // 支持 -i 传入文件夹或文件名（可用 # 分隔多个）；文件夹自动展开为其中的图片文件。
        std::vector<std::string> picture_path_array;
        for (const std::string& tok : FileHelper::splitStringByCsharp(image_path)) {
            if (std::filesystem::is_directory(tok)) {
                std::vector<std::string> dir_images;
                for (const auto& entry : std::filesystem::directory_iterator(tok)) {
                    if (!entry.is_regular_file()) continue;
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                        ext == ".bmp" || ext == ".tif" || ext == ".tiff" || ext == ".webp") {
                        dir_images.push_back(entry.path().string());
                    }
                }
                std::sort(dir_images.begin(), dir_images.end());
                std::cout << "[INFO] Folder '" << tok << "': "
                          << dir_images.size() << " image(s) found" << std::endl;
                for (auto& p : dir_images) picture_path_array.push_back(std::move(p));
            } else {
                picture_path_array.push_back(tok);
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        json array_result = json::array();

        for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++) {
            cv::Mat src_img = cv::imread(picture_path_array[pic_number]);
            if (!src_img.data) {
                std::cerr << "[ERROR] Cannot read image: " << picture_path_array[pic_number] << std::endl;
                json item;
                item["picture_id"] = pic_number + 1;
                item["state_flag"] = "NG";
                item["zifu_type"] = "";
                item["penma_version"] = "new";
                item["result"] = "";
                item["picture_path"] = "";
                item["duanmian"] = "no";
                array_result.push_back(item);
                continue;
            }

            Pipeline::GxJingzhengPipelineResult pres =
                pipeline_->process(src_img, station_id, heat_number, true);

            json item;
            item["picture_id"] = pic_number + 1;
            item["state_flag"] = pres.state_flag;
            item["zifu_type"] = pres.zifu_type;
            item["penma_version"] = pres.penma_version;
            item["result"] = pres.ocr_text;
            item["picture_path"] = "";
            item["duanmian"] = pres.duanmian;
            array_result.push_back(item);

            if (!pres.annotated_image.empty()) {
                std::string save_path = "gx_jingzheng_result_" + std::to_string(pic_number) + ".jpg";
                cv::imwrite(save_path, pres.annotated_image);
                std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
            }

            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);
            saveCharCrops(pres, pic_number, t);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        json root_all;
        root_all["station_id"] = station_id;
        root_all["all_results"] = array_result;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root_all.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    // zifu 分支：保存每个字符的方向矫正后裁剪图（OCR 输入）到 char_crop_dir，扁平命名：
    // <char_crop_dir>\YYYYMMDDHHMMSS_p<pic>_c<char>.jpg
    // char_crop_dir 为空或非 zifu 分支时跳过（gangbiao 分支无 chars）
    void saveCharCrops(const Pipeline::GxJingzhengPipelineResult& result,
                       int pic_number, const tm* t)
    {
        if (config_.char_crop_dir.empty()) return;
        if (result.branch != config_.zifu_class_name) return;
        FileHelper::ensureDirectoryExists(config_.char_crop_dir);
        for (int ci = 0; ci < (int)result.chars.size(); ci++) {
            const cv::Mat& crop = result.chars[ci].image_after_flip;
            if (crop.empty()) continue;
            std::string save_name = cv::format("%s\\%d%02d%02d%02d%02d%02d_p%d_c%d.jpg",
                config_.char_crop_dir.c_str(),
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec,
                pic_number, ci + 1);
            cv::imwrite(save_name, crop);
        }
    }

    GxJingzhengServerConfig config_;
    std::unique_ptr<Pipeline::GxJingzhengPipeline> pipeline_;
    std::mutex mtx_;
};

GxJingzhengService::GxJingzhengService(const std::string& config_path)
    : m_pHandle(std::make_shared<GxJingzhengServicePrivate>(config_path))
{
}

GxJingzhengService::~GxJingzhengService() = default;

const GxJingzhengServerConfig& GxJingzhengService::config() const
{
    return m_pHandle->config();
}

std::string GxJingzhengService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int GxJingzhengService::runLocalTest(const std::string& image_path,
                                      const std::string& heat_number,
                                      int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
