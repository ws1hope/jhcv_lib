#include "JHDeepCore.h"
#include "tiangangfujian_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class TiangangfujianServicePrivate {
public:
    explicit TiangangfujianServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadTiangangfujianConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::TiangangfujianPipeline>(config_);
    }

    const TiangangfujianServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Tiangangfujian.txt";
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
            picture_path = req_json.value("picture_path", "");
            if (req_json.contains("station_id")) {
                const auto& v = req_json["station_id"];
                station_id = v.is_string() ? std::stoi(v.get<std::string>()) : v.get<int>();
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path << std::endl;

        // 单图：取 '#' 分隔后的第一个路径
        std::vector<std::string> path_array = FileHelper::splitStringByCsharp(picture_path);
        std::string first_path = path_array.empty() ? picture_path : path_array[0];

        json root = buildResult(first_path, station_id, false, fout, "");

        std::cout << "[RESULT] " << root.dump() << std::endl;
        fout.close();

        return root.dump();
    }

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     int station_id,
                     const std::string& out_dir)
    {
        // 本地测试结果输出目录：未指定时用当前目录下的 tiangangfujian_result/
        std::string local_out_dir = out_dir.empty() ? "tiangangfujian_result" : out_dir;
        FileHelper::ensureDirectoryExists(local_out_dir);

        std::cout << "=== Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << "  out_dir: " << local_out_dir << std::endl;
        std::cout << std::endl;

        std::vector<std::string> path_array = FileHelper::splitStringByCsharp(image_path);
        std::string first_path = path_array.empty() ? image_path : path_array[0];

        auto start = std::chrono::high_resolution_clock::now();
        std::ofstream fout;  // 本地测试不写日志
        json root = buildResult(first_path, station_id, true, fout, local_out_dir);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        // 读取失败（图片不存在/不可解码）返回 1，便于批量测试汇总
        return (root.value("read_picture_flag", "") == "NG") ? 1 : 0;
    }

private:
    // 单图处理，返回扁平 JSON（沿用 xintiangang 协议）：
    //   图像解析失败 -> read_picture_flag=NG, rec_state_flag=NG, all_results=""
    //   解析成功有结果 -> read_picture_flag=OK, rec_state_flag=OK, all_results=OCR 文本
    //   解析成功无结果 -> read_picture_flag=OK, rec_state_flag=NG, all_results=""
    json buildResult(const std::string& picture_path,
                     int station_id, bool verbose,
                     std::ofstream& fout,
                     const std::string& out_dir)
    {
        json root;
        root["station_id"] = std::to_string(station_id);

        cv::Mat src_img = cv::imread(picture_path);
        if (!src_img.data) {
            root["read_picture_flag"] = "NG";
            root["rec_state_flag"] = "NG";
            root["all_results"] = "";
            root["picture_path"] = picture_path;
            fout << "detect failed! empty image: " << picture_path << std::endl;
            return root;
        }

        Pipeline::TiangangfujianPipelineResult pr = pipeline_->process(src_img, verbose);

        // 结果图保存路径：本地测试用 out_dir，服务端用 result_dir
        std::string save_path;
        std::string stem;
        if (verbose) {
            stem = std::filesystem::path(picture_path).stem().string();
            if (stem.empty()) stem = "result";
            save_path = out_dir + "/" + stem + "_result.jpg";
        } else {
            time_t currtime = time(NULL);
            tm* t = localtime(&currtime);
            std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            FileHelper::ensureDirectoryExists(folderPath);

            save_path = cv::format(
                "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d.jpg",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec);
        }

        if (!pr.annotated_image.empty()) {
            cv::imwrite(save_path, pr.annotated_image);
        } else {
            cv::imwrite(save_path, src_img);
        }
        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }

        // 保存断面框裁剪图（section_crop_dir 为空则不保存）
        if (!config_.section_crop_dir.empty() && !pr.sections.empty()) {
            for (int i = 0; i < (int)pr.sections.size(); i++) {
                if (pr.sections[i].section_image.empty()) continue;
                std::string crop_path;
                if (verbose) {
                    crop_path = cv::format("%s/%s_sec_%02d.jpg",
                        out_dir.c_str(), stem.c_str(), i + 1);
                } else {
                    time_t currtime = time(NULL);
                    tm* t = localtime(&currtime);
                    std::string cropFolder = cv::format("%s\\station_%02d\\%d%02d%02d",
                        config_.section_crop_dir.c_str(), station_id,
                        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
                    FileHelper::ensureDirectoryExists(cropFolder);
                    crop_path = cv::format(
                        "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_sec_%02d.jpg",
                        config_.section_crop_dir.c_str(), station_id,
                        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                        t->tm_hour, t->tm_min, t->tm_sec, i + 1);
                }
                cv::imwrite(crop_path, pr.sections[i].section_image);
                if (verbose) {
                    std::cout << "[INFO] Section crop saved: " << crop_path << std::endl;
                } else {
                    fout << "section crop saved: " << crop_path << std::endl;
                }
            }
        }

        // 保存方向矫正后字符裁剪图（char_crop_dir 为空则不保存）
        if (!config_.char_crop_dir.empty()) {
            int char_idx = 0;
            for (int i = 0; i < (int)pr.sections.size(); i++) {
                for (int ci = 0; ci < (int)pr.sections[i].chars.size(); ci++) {
                    const auto& chr = pr.sections[i].chars[ci];
                    if (chr.image_after_flip.empty()) continue;
                    std::string crop_path;
                    if (verbose) {
                        crop_path = cv::format("%s/%s_sec_%02d_char_%02d.jpg",
                            out_dir.c_str(), stem.c_str(), i + 1, ci + 1);
                    } else {
                        time_t currtime = time(NULL);
                        tm* t = localtime(&currtime);
                        std::string cropFolder = cv::format("%s\\station_%02d\\%d%02d%02d",
                            config_.char_crop_dir.c_str(), station_id,
                            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
                        FileHelper::ensureDirectoryExists(cropFolder);
                        crop_path = cv::format(
                            "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_sec_%02d_char_%02d.jpg",
                            config_.char_crop_dir.c_str(), station_id,
                            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                            t->tm_hour, t->tm_min, t->tm_sec, i + 1, ci + 1);
                    }
                    cv::imwrite(crop_path, chr.image_after_flip);
                    char_idx++;
                    if (verbose) {
                        std::cout << "[INFO] Char crop saved: " << crop_path << std::endl;
                    } else {
                        fout << "char crop saved: " << crop_path << std::endl;
                    }
                }
            }
            (void)char_idx;
        }

        root["read_picture_flag"] = "OK";

        if (!pr.all_results.empty()) {
            root["rec_state_flag"] = "OK";
            root["all_results"] = pr.all_results;
        } else {
            root["rec_state_flag"] = "NG";
            root["all_results"] = "";
        }

        root["picture_path"] = save_path;

        return root;
    }

    TiangangfujianServerConfig config_;
    std::unique_ptr<Pipeline::TiangangfujianPipeline> pipeline_;
    std::mutex mtx_;
};

TiangangfujianService::TiangangfujianService(const std::string& config_path)
    : m_pHandle(std::make_shared<TiangangfujianServicePrivate>(config_path))
{
}

TiangangfujianService::~TiangangfujianService() = default;

const TiangangfujianServerConfig& TiangangfujianService::config() const
{
    return m_pHandle->config();
}

std::string TiangangfujianService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int TiangangfujianService::runLocalTest(const std::string& image_path,
                                        const std::string& heat_number,
                                        int station_id,
                                        const std::string& out_dir)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id, out_dir);
}

} // namespace JHDeepCore
