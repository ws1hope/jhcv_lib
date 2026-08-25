#include "JHDeepCore.h"
#include "fujian_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <mutex>

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

class FujianServicePrivate {
public:
    explicit FujianServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadFujianConfig(config_path))
    {
    }

    const FujianServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Fujian.txt";
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

        json root = buildResult(first_path, station_id, heat_number, false, fout);

        std::cout << "[RESULT] " << root.dump() << std::endl;
        fout.close();

        return root.dump();
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

        std::vector<std::string> path_array = FileHelper::splitStringByCsharp(image_path);
        std::string first_path = path_array.empty() ? image_path : path_array[0];

        auto start = std::chrono::high_resolution_clock::now();
        std::ofstream fout;  // 本地测试不写日志
        json root = buildResult(first_path, station_id, heat_number, true, fout);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root.dump(2) << std::endl;
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

private:
    // heat_num 定位数字决定 roi：10 位取倒数第 2 位，11 位取倒数第 3 位；
    // 数字 1-N -> rois[N-1]，N 为配置的 ROI 数量。其他长度/数字返回 -1
    static int selectRoiIndex(const std::string& heat_number, size_t roi_count)
    {
        size_t len = heat_number.size();
        if (len != 10 && len != 11) return -1;
        size_t pos = (len == 10) ? len - 2 : len - 3;
        char c = heat_number[pos];
        if (c < '1' || c > '9') return -1;
        if ((size_t)(c - '0') > roi_count) return -1;
        return c - '0';
    }

    // 工位 pipeline 懒加载：首次请求该工位时加载并缓存（handleRequest 大锁保护）
    Pipeline::FujianPipeline* getPipeline(int station_id, const FujianStationConfig& st)
    {
        auto it = pipelines_.find(station_id);
        if (it != pipelines_.end()) return it->second.get();

        auto pipeline = std::make_unique<Pipeline::FujianPipeline>(st, config_.device);
        Pipeline::FujianPipeline* raw = pipeline.get();
        pipelines_[station_id] = std::move(pipeline);
        return raw;
    }

    // 单图处理，返回标准嵌套 JSON：
    //   {station_id, all_results: [{picture_id=1, state_flag, zifu_type="Penma",
    //                               result, picture_path}]}
    //   工位未配置 / heat_num 无法定位 roi / 图像解析失败 / OCR 无结果
    //     -> state_flag=NG, result=""，picture_path 仍填路径
    json buildResult(const std::string& picture_path,
                     int station_id, const std::string& heat_number,
                     bool verbose, std::ofstream& fout)
    {
        json root;
        root["station_id"] = station_id;

        json item;
        item["picture_id"] = 1;
        item["zifu_type"] = "Penma";

        // NG：result 置空，picture_path 填路径
        auto ngResult = [&](const std::string& path) {
            item["state_flag"] = "NG";
            item["result"] = "";
            item["picture_path"] = path;
            root["all_results"] = json::array({item});
            return root;
        };

        // config 只配置一个工位：直接取第一个，不按 station_id 匹配
        if (config_.stations.empty()) {
            fout << "no station configured" << std::endl;
            return ngResult(picture_path);
        }
        const FujianStationConfig* st = &config_.stations[0];

        int roi_index = selectRoiIndex(heat_number, st->rois.size());
        if (roi_index < 0) {
            fout << "cannot select roi from heat_number: " << heat_number
                 << " (roi count: " << st->rois.size() << ")" << std::endl;
            return ngResult(picture_path);
        }
        const std::array<int, 4>& roi_arr = st->rois[roi_index - 1];
        cv::Rect roi(roi_arr[0], roi_arr[1], roi_arr[2], roi_arr[3]);

        cv::Mat src_img = cv::imread(picture_path);
        if (!src_img.data) {
            fout << "detect failed! empty image: " << picture_path << std::endl;
            return ngResult(picture_path);
        }

        // pipeline 缓存按配置工位的 station_id（配置只有一个工位，避免不同请求
        // station_id 重复加载同一份模型）
        Pipeline::FujianPipeline* pipeline = getPipeline(st->station_id, *st);
        Pipeline::FujianPipelineResult pr = pipeline->process(src_img, roi, roi_index,
                                                              heat_number, verbose);

        time_t currtime = time(NULL);
        tm* t = localtime(&currtime);

        // 保存 ROI 裁剪图与字符裁剪图（目录为空时跳过）
        saveRoiCrop(pr, station_id, t);
        saveCharCrops(pr, station_id, t);

        // 结果图保存路径：本地测试用当前目录，服务端用 result_dir，文件名 时间戳_炉号.jpg
        std::string save_path;
        if (verbose) {
            save_path = "fujian_result.jpg";
        } else {
            std::string folderPath = cv::format("%s\\station_%02d\\%d%02d%02d",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
            FileHelper::ensureDirectoryExists(folderPath);

            save_path = cv::format(
                "%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_%s.jpg",
                config_.result_dir.c_str(), station_id,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec, heat_number.c_str());
        }

        if (!pr.annotated_image.empty()) {
            cv::imwrite(save_path, pr.annotated_image);
        } else {
            cv::imwrite(save_path, src_img);
        }

        item["state_flag"] = pr.full_text.empty() ? "NG" : "OK";
        // result 加 '#' 分隔：11 位在第 6 位后，其他在第 5 位后
        std::string result_text = pr.full_text;
        if (result_text.size() > 5) {
            result_text.insert(result_text.size() == 11 ? 6 : 5, "#");
        }
        item["result"] = result_text;
        item["picture_path"] = save_path;
        root["all_results"] = json::array({item});

        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }
        return root;
    }

    // 保存 ROI 裁剪图（送分割的输入）到 roi_crop_dir，扁平命名：
    // <roi_crop_dir>\YYYYMMDDHHMMSS_s<station>_roi<index>.jpg
    // roi_crop_dir 为空时跳过
    void saveRoiCrop(const Pipeline::FujianPipelineResult& result, int station_id, const tm* t)
    {
        if (config_.roi_crop_dir.empty()) return;
        if (result.roi_image.empty()) return;
        FileHelper::ensureDirectoryExists(config_.roi_crop_dir);
        std::string save_name = cv::format("%s\\%d%02d%02d%02d%02d%02d_s%02d_roi%d.jpg",
            config_.roi_crop_dir.c_str(),
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec,
            station_id, result.roi_index);
        cv::imwrite(save_name, result.roi_image);
    }

    // 保存每个字符的裁剪图（OCR 输入）到 char_crop_dir，扁平命名：
    // <char_crop_dir>\YYYYMMDDHHMMSS_s<station>_roi<index>_c<char>.jpg
    // char_crop_dir 为空时跳过
    void saveCharCrops(const Pipeline::FujianPipelineResult& result, int station_id, const tm* t)
    {
        if (config_.char_crop_dir.empty()) return;
        FileHelper::ensureDirectoryExists(config_.char_crop_dir);
        for (int ci = 0; ci < (int)result.chars.size(); ci++) {
            const cv::Mat& crop = result.chars[ci].image;
            if (crop.empty()) continue;
            std::string save_name = cv::format("%s\\%d%02d%02d%02d%02d%02d_s%02d_roi%d_c%d.jpg",
                config_.char_crop_dir.c_str(),
                t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                t->tm_hour, t->tm_min, t->tm_sec,
                station_id, result.roi_index, ci + 1);
            cv::imwrite(save_name, crop);
        }
    }

    FujianServerConfig config_;
    std::map<int, std::unique_ptr<Pipeline::FujianPipeline>> pipelines_;  // 工位懒加载缓存
    std::mutex mtx_;
};

FujianService::FujianService(const std::string& config_path)
    : m_pHandle(std::make_shared<FujianServicePrivate>(config_path))
{
}

FujianService::~FujianService() = default;

const FujianServerConfig& FujianService::config() const
{
    return m_pHandle->config();
}

std::string FujianService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int FujianService::runLocalTest(const std::string& image_path,
                                const std::string& heat_number,
                                int station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
