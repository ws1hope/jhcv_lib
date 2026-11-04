#include "JHDeepCore.h"
#include "huaxin_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using json = nlohmann::json;

namespace JHDeepCore {

namespace {

int64_t nowMilliseconds()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct MotionState {
    bool has_anchor = false;
    bool has_last_center = false;
    cv::Point2f anchor;
    cv::Point2f last_center;
    std::string heat;
    cv::Size image_size;
    int64_t last_seen_ms = 0;
};

// 单帧运动判定结果，附带给结果图可视化用的坐标
struct MotionDecision {
    std::string state = "still";
    bool updated = false;       // 本帧是否真正做了运动判定
    bool reset = false;         // 本帧是否发生状态重置
    bool has_reference = false; // 是否有可画的判基准点
    cv::Point2f reference;      // 本次比较所用的 anchor（更新前）
    cv::Point2f current;        // 当前最左框中心
};

void putTextOutlined(cv::Mat& image, const std::string& text, cv::Point org,
                     const cv::Scalar& color, double scale = 1.2, int thickness = 2)
{
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    org.x = std::max(4, std::min(org.x, image.cols - ts.width - 4));
    org.y = std::max(ts.height + 4, std::min(org.y, image.rows - baseline - 4));
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 0), thickness + 3, cv::LINE_AA);
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv::LINE_AA);
}

// 在结果图上标注判基准点(REF)、当前中心点(CUR)和位移
void drawMotionDebug(cv::Mat& image, const MotionDecision& decision)
{
    if (image.empty()) {
        return;
    }

    cv::Scalar state_color(0, 255, 255);
    if (decision.state == "forward") {
        state_color = cv::Scalar(0, 255, 0);
    } else if (decision.state == "back") {
        state_color = cv::Scalar(0, 0, 255);
    }

    const cv::Point cur(cvRound(decision.current.x), cvRound(decision.current.y));

    if (decision.has_reference) {
        const cv::Point ref(cvRound(decision.reference.x), cvRound(decision.reference.y));

        if (cv::norm(cur - ref) >= 2) {
            cv::arrowedLine(image, ref, cur, cv::Scalar(0, 0, 0), 8, cv::LINE_AA, 0, 0.25);
            cv::arrowedLine(image, ref, cur, cv::Scalar(255, 255, 255), 4, cv::LINE_AA, 0, 0.25);
        }

        cv::circle(image, ref, 14, cv::Scalar(0, 0, 0), 6, cv::LINE_AA);
        cv::circle(image, ref, 14, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
        cv::drawMarker(image, ref, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
        putTextOutlined(image, "REF", ref + cv::Point(18, -18), cv::Scalar(255, 0, 255));

        float dx = decision.current.x - decision.reference.x;
        float dy = decision.current.y - decision.reference.y;
        std::string label = cv::format("dx=%.1f dy=%.1f %s", dx, dy, decision.state.c_str());
        putTextOutlined(image, label,
                        cv::Point((ref.x + cur.x) / 2 + 12, (ref.y + cur.y) / 2 - 12),
                        state_color);
    }

    cv::circle(image, cur, 10, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    cv::circle(image, cur, 8, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
    putTextOutlined(image, decision.reset ? "CUR (RESET)" : "CUR",
                    cur + cv::Point(14, 30), cv::Scalar(255, 255, 0));
}

} // namespace

class HuaxinServicePrivate {
public:
    explicit HuaxinServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadHuaxinConfig(config_path))
    {
        pipeline_ = std::make_unique<Pipeline::HuaxinPipeline>(config_);
    }

    const HuaxinServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Huaxin.txt";
        std::string fileFullName = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(fileFullName.c_str(), std::ios::app);

        std::string station_id = "0";
        std::string heat_number;
        std::string picture_path;

        try {
            json req_json = json::parse(req_body);
            heat_number = req_json.value("heat_number", "");
            picture_path = req_json.value("picture_path", "");
            if (req_json.contains("station_id")) {
                const auto& v = req_json["station_id"];
                if (v.is_string()) {
                    station_id = v.get<std::string>();
                } else if (v.is_number_integer()) {
                    station_id = std::to_string(v.get<int>());
                }
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        fout << "recv station:" << station_id
             << " heat:" << heat_number
             << " path:" << picture_path << std::endl;

        // 单张图片：picture_path 整体作为图片路径，不按任何字符拆分
        json root;
        root["station_id"] = station_id;
        json all_results = json::array();
        double total_inference_ms = 0.0;
        all_results.push_back(
            buildResult(picture_path, station_id, heat_number, 1, false,
                        nowMilliseconds(), fout, total_inference_ms));
        root["all_results"] = all_results;

        root["time_cost"] = std::round(total_inference_ms * 100.0) / 100.0;

        std::cout << "[RESULT] " << root.dump() << std::endl;
        fout.close();

        return root.dump();
    }

    int runLocalTest(const std::string& image_path,
                     const std::string& heat_number,
                     const std::string& station_id)
    {
        std::cout << "=== Local Test Mode ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  station: " << station_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;
        std::cout << std::endl;

        std::ofstream fout;  // 本地测试不写日志

        json root;
        root["station_id"] = station_id;
        json all_results = json::array();
        double total_inference_ms = 0.0;
        all_results.push_back(
            buildResult(image_path, station_id, heat_number, 1, true,
                        nowMilliseconds(), fout, total_inference_ms));
        root["all_results"] = all_results;

        root["time_cost"] = std::round(total_inference_ms * 100.0) / 100.0;

        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        std::cout << root.dump(2) << std::endl;
        std::cout << "inference time: " << total_inference_ms << " ms" << std::endl;

        return 0;
    }

private:
    MotionDecision updateMotion(const std::string& station_id,
                                const std::string& heat_number,
                                const cv::Point2f& center,
                                const cv::Size& image_size,
                                int64_t now_ms, bool verbose,
                                std::ofstream& fout)
    {
        MotionDecision decision;
        if (!config_.motion_enable) {
            return decision;
        }

        decision.updated = true;
        decision.current = center;

        MotionState& state = motion_states_[station_id];

        bool reset = false;
        if (state.has_anchor) {
            if (state.heat != heat_number ||
                state.image_size != image_size ||
                now_ms - state.last_seen_ms > config_.motion_reset_timeout_ms) {
                reset = true;
            } else if (state.has_last_center) {
                float jump_x = center.x - state.last_center.x;
                float jump_y = center.y - state.last_center.y;
                float jump = std::sqrt(jump_x * jump_x + jump_y * jump_y);
                if (jump > config_.motion_max_jump_px) {
                    reset = true;
                }
            }
        }
        if (reset) {
            decision.reset = true;
            state = MotionState{};
        }

        state.heat = heat_number;
        state.image_size = image_size;
        state.last_seen_ms = now_ms;

        if (!state.has_anchor) {
            state.has_anchor = true;
            state.has_last_center = true;
            state.anchor = center;
            state.last_center = center;
            return decision;
        }

        decision.reference = state.anchor;
        decision.has_reference = true;

        float dx = center.x - state.anchor.x;
        float dy = center.y - state.anchor.y;
        int min_move = config_.motion_min_move_px > 0 ? config_.motion_min_move_px : 1;

        std::string motion = "still";
        if (std::fabs(dx) >= static_cast<float>(min_move)) {
            motion = (dx < 0.0f) ? "forward" : "back";
            state.anchor = center;
        }
        state.last_center = center;
        state.has_last_center = true;

        if (verbose) {
            std::cout << "[DEBUG] motion station=" << station_id
                      << " dx=" << dx << " dy=" << dy
                      << " min_move=" << min_move
                      << " -> " << motion << std::endl;
        }
        if (fout.is_open()) {
            fout << "motion station:" << station_id
                 << " heat:" << heat_number
                 << " dx:" << dx << " dy:" << dy
                 << " result:" << motion << std::endl;
        }

        decision.state = motion;
        return decision;
    }

    // 单图处理，返回 all_results 数组中的一项：
    //   state_flag   - 整体是否成功 OK/NG（detail_state 为 OK 时为 OK）
    //   detail_state - OK 或 NG 原因码：
    //                  NO_LABEL / NO_GBNUM / INVALID_EXT / PATH_ERROR /
    //                  READ_ERROR / RECOGNIZE_ERROR / SAVE_ERROR
    //   result       - 成功时为铸坯端面识别结果，失败时固定为 ""
    //   picture_path - 结果图保存路径（输入图目录\results\station_<id>\<OK|NG>\原文件名）
    //   motion       - 最左框运动方向：forward/back/still
    json buildResult(const std::string& picture_path,
                     const std::string& station_id,
                     const std::string& heat_number,
                     int picture_id, bool verbose,
                     int64_t now_ms,
                     std::ofstream& fout,
                     double& total_inference_ms)
    {
        json item;
        item["picture_id"] = std::to_string(picture_id);
        item["result"] = "";
        item["motion"] = "still";

        auto setFailure = [&item](const std::string& code) {
            item["state_flag"] = "NG";
            item["detail_state"] = code;
            item["result"] = "";
        };

        // 1. 图片后缀白名单（不区分大小写）
        static const std::vector<std::string> kValidExts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"};
        std::string ext = std::filesystem::path(picture_path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (std::find(kValidExts.begin(), kValidExts.end(), ext) == kValidExts.end()) {
            setFailure("INVALID_EXT");
            item["picture_path"] = picture_path;
            fout << "invalid extension: " << picture_path << std::endl;
            return item;
        }

        // 2. 路径必须存在且为普通文件
        std::error_code fs_ec;
        if (!std::filesystem::exists(picture_path, fs_ec) ||
            !std::filesystem::is_regular_file(picture_path, fs_ec)) {
            setFailure("PATH_ERROR");
            item["picture_path"] = picture_path;
            fout << "path not found or not a file: " << picture_path << std::endl;
            return item;
        }

        // 3. 图片读取/解码
        cv::Mat src_img = cv::imread(picture_path);
        if (!src_img.data) {
            setFailure("READ_ERROR");
            item["picture_path"] = picture_path;
            fout << "detect failed! empty image: " << picture_path << std::endl;
            return item;
        }

        // 4. 模型推理/识别
        Pipeline::HuaxinPipelineResult pr;
        try {
            pr = pipeline_->process(src_img, station_id, verbose);
        } catch (const std::exception& e) {
            setFailure("RECOGNIZE_ERROR");
            item["picture_path"] = picture_path;
            fout << "recognize error: " << e.what() << std::endl;
            return item;
        } catch (...) {
            setFailure("RECOGNIZE_ERROR");
            item["picture_path"] = picture_path;
            fout << "recognize error: unknown exception" << std::endl;
            return item;
        }
        total_inference_ms += pr.inference_time_ms;

        // 只有最左框仍在有效检测范围内时才更新运动方向状态
        std::string motion = "still";
        MotionDecision decision;
        if (pr.leftmost_index >= 0 && !pr.det2_skipped && pr.leftmost_roi.area() > 0) {
            cv::Point2f center(pr.leftmost_roi.x + pr.leftmost_roi.width / 2.0f,
                               pr.leftmost_roi.y + pr.leftmost_roi.height / 2.0f);
            decision = updateMotion(station_id, heat_number, center, src_img.size(),
                                    now_ms, verbose, fout);
            motion = decision.state;
        }
        item["motion"] = motion;

        // 5. 识别结果判定
        bool ok = !pr.all_results.empty();
        if (ok) {
            item["state_flag"] = "OK";
            item["detail_state"] = "OK";
            item["result"] = pr.all_results;
        } else {
            setFailure(pr.failure_reason.empty() ? "NO_LABEL" : pr.failure_reason);
        }

        // 在结果图上标注运动方向
        if (!pr.annotated_image.empty()) {
            std::string label = "STILL";
            cv::Scalar color(0, 255, 255);
            if (motion == "forward") {
                label = "FORWARD";
                color = cv::Scalar(0, 255, 0);
            } else if (motion == "back") {
                label = "BACK";
                color = cv::Scalar(0, 0, 255);
            }
            cv::putText(pr.annotated_image, label, cv::Point(10, 240),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, color, 5);

            // 本次判基准点、当前中心点及位移
            if (decision.updated) {
                drawMotionDebug(pr.annotated_image, decision);
            }
        }

        // 6. 结果图保存路径：输入图所在目录\results\station_<工位>\<OK|NG>\原文件名；
        //    本地测试直接存当前目录。目录名只取决于整体是否成功，不使用 detail_state 文本。
        std::string save_path;
        if (verbose) {
            save_path = "huaxin_result.jpg";
        } else {
            size_t pos = picture_path.find_last_of("\\/");
            std::string dir = (pos == std::string::npos) ? "." : picture_path.substr(0, pos);
            std::string name = (pos == std::string::npos) ? picture_path : picture_path.substr(pos + 1);
            std::string folder = dir + "\\results\\station_" + station_id
                                 + "\\" + (ok ? "OK" : "NG");
            std::error_code dir_ec;
            std::filesystem::create_directories(folder, dir_ec);
            if (dir_ec || !std::filesystem::is_directory(folder, dir_ec)) {
                setFailure("SAVE_ERROR");
                item["picture_path"] = picture_path;
                fout << "create dir failed: " << folder << std::endl;
                return item;
            }
            save_path = folder + "\\" + name;
        }

        const cv::Mat& out_img = pr.annotated_image.empty() ? src_img : pr.annotated_image;
        if (!cv::imwrite(save_path, out_img)) {
            setFailure("SAVE_ERROR");
            item["picture_path"] = picture_path;
            fout << "imwrite failed: " << save_path << std::endl;
            return item;
        }

        item["picture_path"] = save_path;

        if (verbose) {
            std::cout << "[INFO] Annotated image saved: " << save_path << std::endl;
        }
        return item;
    }

    HuaxinServerConfig config_;
    std::unique_ptr<Pipeline::HuaxinPipeline> pipeline_;
    std::unordered_map<std::string, MotionState> motion_states_;
    std::mutex mtx_;
};

HuaxinService::HuaxinService(const std::string& config_path)
    : m_pHandle(std::make_shared<HuaxinServicePrivate>(config_path))
{
}

HuaxinService::~HuaxinService() = default;

const HuaxinServerConfig& HuaxinService::config() const
{
    return m_pHandle->config();
}

std::string HuaxinService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int HuaxinService::runLocalTest(const std::string& image_path,
                                const std::string& heat_number,
                                const std::string& station_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, station_id);
}

} // namespace JHDeepCore
