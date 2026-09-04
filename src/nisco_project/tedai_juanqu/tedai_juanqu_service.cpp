#include "JHDeepCore.h"
#include "tedai_juanqu_pipeline.h"
#include "reel_types.h"
#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include <vector>

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace JHDeepCore {

namespace {

// 旧服务相机编号范围（begin_cam_num/end_cam_num）
const int kBeginCamNum = 0;
const int kEndCamNum = 5;

// multipart metadata 中一条图片信息
struct ReelRequestItem {
    int camera_id = -1;
    std::string file_key;
};

// 解析 metadata JSON：station_id + pictures_information[{camera_id, file_key}]
// 只保留 files 中实际存在的 file_key（与旧服务一致）
bool parseMetadata(const std::string &metadata_json,
                   const std::map<std::string, std::string> &files,
                   std::string &station_id, std::vector<ReelRequestItem> &items,
                   std::ostream &log)
{
    json root = json::parse(metadata_json, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        log << "metadata parse failed" << std::endl;
        return false;
    }
    if (root.contains("station_id") && root["station_id"].is_string()) {
        station_id = root["station_id"].get<std::string>();
    }
    if (!root.contains("pictures_information") ||
        !root["pictures_information"].is_array()) {
        return true;  // 无图片信息，调用方按空列表处理
    }
    for (const auto &item : root["pictures_information"]) {
        if (!item.is_object()) continue;
        if (!item.contains("camera_id") || !item["camera_id"].is_number()) continue;
        if (!item.contains("file_key") || !item["file_key"].is_string()) continue;

        ReelRequestItem ri;
        ri.camera_id = item["camera_id"].get<int>();
        ri.file_key = item["file_key"].get<std::string>();
        if (files.find(ri.file_key) == files.end()) {
            log << "file_key not found: " << ri.file_key << std::endl;
            continue;
        }
        items.push_back(ri);
    }
    return true;
}

// 按旧协议组装单相机结果：
// {camera_id, read_picture_flag, result_picture_path,
//  single_camera_results: [{roi_id, roi_boxCount,
//   panjuan_location_information: [{panjuan_id,x,y,width,height,location}]}]}
void appendCameraResult(ojson &all_results, const Pipeline::ReelCameraOutput &cam)
{
    ojson root2;
    root2["camera_id"] = cam.camera_id;
    root2["read_picture_flag"] = cam.read_picture_flag;
    root2["result_picture_path"] = cam.result_pic_path;

    ojson single_camera_results = ojson::array();
    const struct {
        const char *roi_id;
        const std::vector<Pipeline::ReelRect> *rects;
    } rois[] = {
        {"inside", &cam.inside},
        {"outside", &cam.outside},
        {"collect", &cam.collect},
    };
    for (const auto &roi : rois) {
        ojson root3;
        root3["roi_id"] = roi.roi_id;
        root3["roi_boxCount"] = (int)roi.rects->size();
        ojson boxes = ojson::array();
        for (size_t n = 0; n < roi.rects->size(); n++) {
            ojson root4;
            root4["panjuan_id"] = (int)n;
            root4["x"] = (*roi.rects)[n].x;
            root4["y"] = (*roi.rects)[n].y;
            root4["width"] = (*roi.rects)[n].width;
            root4["height"] = (*roi.rects)[n].height;
            root4["location"] = (*roi.rects)[n].location;
            boxes.push_back(root4);
        }
        root3["panjuan_location_information"] = boxes;
        single_camera_results.push_back(root3);
    }
    root2["single_camera_results"] = single_camera_results;
    all_results.push_back(root2);
}

// 分步耗时明细 -> 逐行文本（stdout 与请求日志共用）
std::string formatTimingLines(const std::vector<Pipeline::ReelStepTiming> &timings)
{
    std::string out;
    for (const auto &st : timings) {
        out += cv::format(
            "[TIME][cam=%d] clone=%.2f detect=%.2f classify=%.2f "
            "assign=%.2f sort=%.2f dedup=%.2f transform=%.2f draw=%.2f "
            "save=%.2f total=%.2f ms\n",
            st.camera_id, st.clone_ms, st.detect_ms, st.classify_ms,
            st.post_assign_ms, st.post_sort_ms, st.post_dedup_ms,
            st.post_transform_ms, st.post_draw_ms, st.save_ms, st.total_ms);
    }
    return out;
}

} // namespace

class TedaiJuanquServicePrivate {
  public:
    explicit TedaiJuanquServicePrivate(const std::string &config_path)
        : config_(FileHelper::loadTedaiJuanquConfig(config_path)),
          pipeline_(std::make_unique<Pipeline::TedaiJuanquPipeline>(config_))
    {
    }

    const TedaiJuanquServerConfig &config() const { return config_; }

    bool warmup() { return pipeline_->warmup(); }

    // metadata_json：multipart 的 metadata 字段内容（JSON 字符串）
    // files：file_key -> 图片二进制
    std::string handleRequest(const std::string &metadata_json,
                              const std::map<std::string, std::string> &files)
    {
        // 与旧服务一致：全局串行处理（std::lock_guard，异常安全）
        std::lock_guard<std::mutex> lock(mtx_);

        std::ofstream fout = openLog();

        std::string station_id;
        std::vector<ReelRequestItem> items;
        bool meta_ok = parseMetadata(metadata_json, files, station_id, items, fout);
        if (meta_ok) {
            fout << "metadata ok, station_id=" << station_id
                 << ", pictures=" << items.size() << std::endl;
        }

        ojson root;
        root["station_id"] = station_id;
        root["read_message_flag"] = "OK";
        root["all_results"] = ojson::array();

        auto log_and_dump = [&]() {
            std::string result = root.dump();
            fout << "response json: " << result << std::endl;
            return result;
        };

        if (!meta_ok || items.empty()) {
            // 旧协议：metadata 解析失败或无有效图片 -> NG + 空 all_results
            root["read_message_flag"] = "NG";
            fout << "no valid pictures, read_message_flag=NG" << std::endl;
            return log_and_dump();
        }

        // 相机编号校验：范围 [0,5] 且不重复
        bool is_error_id = false;
        std::vector<int> camera_ids;
        for (const auto &ri : items) {
            if (ri.camera_id < kBeginCamNum || ri.camera_id > kEndCamNum) {
                is_error_id = true;
            }
            camera_ids.push_back(ri.camera_id);
        }
        std::sort(camera_ids.begin(), camera_ids.end());
        for (size_t i = 1; i < camera_ids.size(); i++) {
            if (camera_ids[i] == camera_ids[i - 1]) {
                is_error_id = true;
            }
        }
        if (is_error_id) {
            root["read_message_flag"] = "NG";
            fout << "invalid camera id, read_message_flag=NG" << std::endl;
            return log_and_dump();
        }

        // 解码图片（解码失败为空 Mat，按旧服务 read_picture_flag=0 处理）
        auto td0 = std::chrono::high_resolution_clock::now();
        std::vector<Pipeline::ReelFrameInput> frames(items.size());
        cv::parallel_for_(cv::Range(0, (int)items.size()), [&](const cv::Range &r) {
            for (int i = r.start; i < r.end; ++i) {
                const std::string &content = files.at(items[i].file_key);
                cv::Mat buf(1, (int)content.size(), CV_8UC1,
                            const_cast<char *>(content.data()));
                frames[i].camera_id = items[i].camera_id;
                frames[i].image = cv::imdecode(buf, cv::IMREAD_COLOR);
            }
        });
        auto td1 = std::chrono::high_resolution_clock::now();
        double decode_ms =
            std::chrono::duration<double, std::milli>(td1 - td0).count();
        std::cout << "[TIME] decode: " << decode_ms << " ms" << std::endl;
        fout << "[TIME] decode: " << decode_ms << " ms" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Pipeline::ReelCameraOutput> results = pipeline_->process(frames);
        auto t1 = std::chrono::high_resolution_clock::now();
        double process_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[TIME] pipeline process: " << process_ms << " ms" << std::endl;
        fout << "[TIME] pipeline process: " << process_ms << " ms" << std::endl;

        std::cout << "[TIME] detect: " << pipeline_->detectMs()
                  << " ms | classify: " << pipeline_->classifyMs() << " ms"
                  << std::endl;
        fout << "[TIME] detect: " << pipeline_->detectMs()
             << " ms | classify: " << pipeline_->classifyMs() << " ms"
             << std::endl;

        // 分步耗时明细（stdout 与请求日志各打一份，便于对比差异）
        std::string timing_lines = formatTimingLines(pipeline_->lastTimings());
        std::cout << timing_lines;
        fout << timing_lines;

        for (const auto &cam : results) {
            appendCameraResult(root["all_results"], cam);
        }

        fout << "detect finished, cameras=" << results.size() << std::endl;
        return log_and_dump();
    }

  private:
    std::ofstream openLog()
    {
        time_t current_time = time(NULL);
        char time_str[256];
        strftime(time_str, sizeof(time_str), "%Y%m%d", localtime(&current_time));
        std::string filename = std::string(time_str) + "@algorithm.txt";
        std::string full_name = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(full_name.c_str(), std::ios::app);
        return fout;
    }

    TedaiJuanquServerConfig config_;
    std::unique_ptr<Pipeline::TedaiJuanquPipeline> pipeline_;
    std::mutex mtx_;
};

TedaiJuanquService::TedaiJuanquService(const std::string &config_path)
    : m_pHandle(std::make_shared<TedaiJuanquServicePrivate>(config_path))
{
}

TedaiJuanquService::~TedaiJuanquService() = default;

const TedaiJuanquServerConfig &TedaiJuanquService::config() const
{
    return m_pHandle->config();
}

std::string TedaiJuanquService::handleRequest(
    const std::string &metadata_json,
    const std::map<std::string, std::string> &files)
{
    return m_pHandle->handleRequest(metadata_json, files);
}

bool TedaiJuanquService::warmup()
{
    return m_pHandle->warmup();
}

} // namespace JHDeepCore
