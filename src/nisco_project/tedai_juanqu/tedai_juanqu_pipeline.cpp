#include "tedai_juanqu_pipeline.h"
#include "reel_rules.h"

#include "json.hpp"
#include "file_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <iostream>

namespace JHDeepCore {
namespace Pipeline {

using json = nlohmann::json;

// 旧版结构体数组容量
static const int kMaxPanjuanCount = 50;

namespace {

std::string fmtFixed(double value, int digits)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", digits, value);
    return std::string(buf);
}

void drawDashedLine(cv::Mat &img, const cv::Point &a, const cv::Point &b,
                    const cv::Scalar &color, int thickness)
{
    double len = cv::norm(b - a);
    if (len < 1.0) return;
    cv::Point2d dir((b.x - a.x) / len, (b.y - a.y) / len);
    const double dash = 14.0;
    for (double s = 0.0; s < len; s += dash * 2.0) {
        double e = std::min(len, s + dash);
        cv::Point p1(cvRound(a.x + dir.x * s), cvRound(a.y + dir.y * s));
        cv::Point p2(cvRound(a.x + dir.x * e), cvRound(a.y + dir.y * e));
        cv::line(img, p1, p2, color, thickness, cv::LINE_AA);
    }
}

// 左下角信息面板（深色底 + 白字）；panel_slot>0 时整体上移，供多区域标注堆叠
void drawInfoPanel(cv::Mat &img, const std::vector<std::string> &lines,
                   int panel_slot = 0)
{
    if (lines.empty()) return;
    const double scale = 0.6;
    const int line_h = 24;
    const int pad = 8;
    int max_w = 0;
    for (const auto &s : lines) {
        int base = 0;
        cv::Size sz = cv::getTextSize(s, cv::FONT_HERSHEY_DUPLEX, scale, 1, &base);
        max_w = std::max(max_w, sz.width);
    }
    int total_h = (int)lines.size() * line_h + pad * 2;
    int x = 10;
    const int slot_h = line_h * 6 + pad * 2 + 12;  // 单面板最大高度 + 间隔
    int y = img.rows - total_h - 10 - panel_slot * slot_h;
    if (y < 0) y = 0;
    cv::Rect bg(x, y, max_w + pad * 2, total_h);
    cv::rectangle(img, bg, cv::Scalar(30, 30, 30), cv::FILLED);
    for (size_t i = 0; i < lines.size(); i++) {
        cv::putText(img, lines[i], cv::Point(x + pad, y + pad + 16 + (int)i * line_h),
                    cv::FONT_HERSHEY_DUPLEX, scale, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    }
}

// 在结果图上标注进入判定：qishi/zhongzhi 轴线、半径圆、上报/未上报的框与信息面板。
// panel_slot 用于同一相机标注多个区域时信息面板向上堆叠。
void drawEnterDebug(cv::Mat &img, int camera_id, const ReelEnterDebug &dbg,
                    int entry_radius_px, double entry_t_ratio, int panel_slot = 0)
{
    const cv::Scalar kAxisColor(255, 0, 255);    // 品红：轴线
    const cv::Scalar kQishiColor(0, 0, 255);     // 红：起点
    const cv::Scalar kZhongzhiColor(255, 0, 0);  // 蓝：终点
    const cv::Scalar kEnterColor(0, 255, 0);     // 绿：本次上报
    const cv::Scalar kRejectColor(150, 150, 150);
    const cv::Scalar kDupColor(0, 255, 255);
    const cv::Scalar kLeftColor(0, 0, 255);      // 红：已离开仍可见的卷
    const cv::Scalar kGateColor(255, 255, 0);    // 青：出口感应线
    const cv::Scalar kMotionColor(180, 180, 180);  // 灰：运动线段
    const cv::Scalar kCrossColor(255, 255, 255);   // 白：触发跨线的运动线段

    auto drawBox = [&](const ReelRect &box, const cv::Scalar &color, int thickness) {
        cv::Rect rect(box.x - box.width / 2, box.y - box.height / 2, box.width,
                      box.height);
        cv::rectangle(img, rect, color, thickness);
    };

    if (dbg.gate.has_axis) {
        cv::Point q(dbg.gate.qishi.x, dbg.gate.qishi.y);
        cv::Point z(dbg.gate.zhongzhi.x, dbg.gate.zhongzhi.y);
        drawDashedLine(img, q, z, kAxisColor, 2);
        cv::circle(img, q, entry_radius_px, kAxisColor, 1, cv::LINE_AA);
        cv::circle(img, q, 6, kQishiColor, cv::FILLED, cv::LINE_AA);
        cv::circle(img, z, 6, kZhongzhiColor, cv::FILLED, cv::LINE_AA);
        cv::putText(img, "qishi", q + cv::Point(8, -8), cv::FONT_HERSHEY_DUPLEX,
                    0.6, kQishiColor, 1, cv::LINE_AA);
        cv::putText(img, "zhongzhi", z + cv::Point(8, -8),
                    cv::FONT_HERSHEY_DUPLEX, 0.6, kZhongzhiColor, 1, cv::LINE_AA);
    }

    // 出口感应线：过 zhongzhi、垂直于轴（leave 判定边界）
    if (dbg.gate.has_gate) {
        cv::Point ga(dbg.gate.gate_a.x, dbg.gate.gate_a.y);
        cv::Point gb(dbg.gate.gate_b.x, dbg.gate.gate_b.y);
        drawDashedLine(img, ga, gb, kGateColor, 2);
        cv::putText(img, "gate",
                    cv::Point(dbg.gate.zhongzhi.x + 10,
                              dbg.gate.zhongzhi.y + 22),
                    cv::FONT_HERSHEY_DUPLEX, 0.6, kGateColor, 1, cv::LINE_AA);
    }

    // 运动线段：各匹配轨迹上一帧->本帧的中心连线；触发跨线的白色加粗
    for (const auto &seg : dbg.motion) {
        const cv::Scalar &color = seg.crossed ? kCrossColor : kMotionColor;
        const int thickness = seg.crossed ? 3 : 1;
        cv::line(img, cv::Point(seg.from.x, seg.from.y),
                 cv::Point(seg.to.x, seg.to.y), color, thickness, cv::LINE_AA);
        cv::circle(img, cv::Point(seg.from.x, seg.from.y), 4, color, -1,
                   cv::LINE_AA);
        cv::circle(img, cv::Point(seg.to.x, seg.to.y), 4, color, -1, cv::LINE_AA);
        if (seg.crossed) {
            cv::putText(img, "cross", cv::Point(seg.to.x + 8, seg.to.y - 10),
                        cv::FONT_HERSHEY_DUPLEX, 0.6, kCrossColor, 2, cv::LINE_AA);
        }
    }

    for (const auto &box : dbg.left_visible) {
        drawBox(box, kLeftColor, 2);
        cv::putText(img, "left", cv::Point(box.x - box.width / 2, box.y + box.height / 2 + 18),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, kLeftColor, 1, cv::LINE_AA);
    }
    for (const auto &box : dbg.duplicated) {
        drawBox(box, kDupColor, 1);
        cv::putText(img, "dup", cv::Point(box.x - box.width / 2, box.y - box.height / 2 - 4),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, kDupColor, 1, cv::LINE_AA);
    }
    for (const auto &box : dbg.rejected) {
        drawBox(box, kRejectColor, 1);
        cv::putText(img, "no-enter",
                    cv::Point(box.x - box.width / 2, box.y - box.height / 2 - 4),
                    cv::FONT_HERSHEY_DUPLEX, 0.5, kRejectColor, 1, cv::LINE_AA);
    }

    if (dbg.has_event) {
        drawBox(dbg.box, kEnterColor, 3);
        std::string rule = (dbg.rule == ReelEnterDebug::RULE_DIST)
                               ? "dist"
                               : ((dbg.rule == ReelEnterDebug::RULE_T)
                                      ? "t"
                                      : "fallback");
        std::string label = "ENTER d=" + fmtFixed(dbg.distance, 0) +
                            " t=" + fmtFixed(dbg.t, 2) + " rule=" + rule;
        cv::putText(img, label,
                    cv::Point(dbg.box.x - dbg.box.width / 2,
                              dbg.box.y - dbg.box.height / 2 - 8),
                    cv::FONT_HERSHEY_DUPLEX, 0.7, kEnterColor, 2, cv::LINE_AA);
    }

    std::string status;
    if (dbg.baseline) {
        status = "enter=NO (baseline)";
    } else if (dbg.has_event) {
        std::string rule = (dbg.rule == ReelEnterDebug::RULE_DIST)
                               ? "dist"
                               : ((dbg.rule == ReelEnterDebug::RULE_T) ? "t"
                                                                       : "fallback");
        status = "enter=YES rule=" + rule + " d=" + fmtFixed(dbg.distance, 0) +
                 " t=" + fmtFixed(dbg.t, 2);
    } else {
        status = "enter=NO (no candidate)";
    }

    std::vector<std::string> lines;
    lines.push_back("cam" + std::to_string(camera_id) + "  new=" +
                    std::to_string(dbg.new_count) + " tracked=" +
                    std::to_string(dbg.tracked_count) + " dup=" +
                    std::to_string(dbg.dup_count) + " left=" +
                    std::to_string(dbg.left_count));
    lines.push_back(status);
    if (dbg.gate.has_axis) {
        lines.push_back("qishi=(" + std::to_string(dbg.gate.qishi.x) + "," +
                        std::to_string(dbg.gate.qishi.y) + ")  zhongzhi=(" +
                        std::to_string(dbg.gate.zhongzhi.x) + "," +
                        std::to_string(dbg.gate.zhongzhi.y) + ")");
    } else {
        lines.push_back("qishi=none (fallback)");
    }
    lines.push_back("radius=" + std::to_string(entry_radius_px) + "  t_max=" +
                    fmtFixed(entry_t_ratio, 2));
    if (dbg.has_event && dbg.event_id.size() > 14) {
        lines.push_back("event=" + dbg.event_id.substr(0, 14));
    }
    drawInfoPanel(img, lines, panel_slot);
}

} // namespace

TedaiJuanquPipeline::TedaiJuanquPipeline(const TedaiJuanquServerConfig &config)
    : config_(config), enter_tracker_(config.events)
{
    if (!loadLiuzhiRois()) {
        std::cerr << "[ERROR] tedai_juanqu: failed to load liuzhi roi file: "
                  << config_.liuzhi_roi_file << std::endl;
    }

    // camera 5 分类开关：开启但未配模型时无效，回退为关闭行为
    classify_enabled_ = config_.enable_classify && !config_.classify_model.empty();
    if (config_.enable_classify && config_.classify_model.empty()) {
        std::cerr << "[WARN] tedai_juanqu: enable_classify=true but classify_model "
                     "is empty, classification disabled"
                  << std::endl;
    }
}

bool TedaiJuanquPipeline::loadLiuzhiRois()
{
    std::ifstream fin(config_.liuzhi_roi_file);
    if (!fin.is_open()) return false;

    json root = json::parse(fin, nullptr, false);
    if (root.is_discarded() || !root.contains("parameters")) return false;

    for (const auto &item : root["parameters"]) {
        if (!item.is_object() || !item.contains("camera_id")) continue;
        int camera_id = item["camera_id"].get<int>();

        std::vector<ReelRoiInfo> rois;
        if (item.contains("liuzhi_rois")) {
            for (const auto &roi_item : item["liuzhi_rois"]) {
                if (!roi_item.is_object()) continue;
                ReelRoiInfo info;
                if (roi_item.contains("liuzhi_id")) {
                    info.liuzhi_id = roi_item["liuzhi_id"].get<int>();
                }
                if (roi_item.contains("contour_points")) {
                    for (const auto &pt : roi_item["contour_points"]) {
                        if (!pt.is_object()) continue;
                        ReelPoint p;
                        p.x = pt.value("x", 0);
                        p.y = pt.value("y", 0);
                        if ((int)info.contour_points.size() < 20) {  // 旧版固定数组 [20]
                            info.contour_points.push_back(p);
                        }
                    }
                }
                if (roi_item.contains("qishi_zhongzhi")) {
                    for (const auto &pt : roi_item["qishi_zhongzhi"]) {
                        if (!pt.is_object()) continue;
                        ReelPoint p;
                        p.x = pt.value("x", 0);
                        p.y = pt.value("y", 0);
                        if ((int)info.qishi_zhongzhi_points.size() < 20) {
                            info.qishi_zhongzhi_points.push_back(p);
                        }
                    }
                }
                rois.push_back(info);
            }
        }
        if (rois.size() < 3) {
            std::cerr << "[WARN] tedai_juanqu: camera " << camera_id
                      << " has only " << rois.size()
                      << " liuzhi rois (need 3: inside/outside/collect)" << std::endl;
        }

        // 旧版映射：rois[0]=输送线内、rois[1]=输送线外、rois[2]=集卷
        ReelRoiParams params;
        if (rois.size() >= 1) {
            params.inside_polygon = rois[0].contour_points;
            params.inside_qishi = rois[0].qishi_zhongzhi_points;
        }
        if (rois.size() >= 2) {
            params.outside_polygon = rois[1].contour_points;
            params.outside_qishi = rois[1].qishi_zhongzhi_points;
        }
        if (rois.size() >= 3) {
            params.collect_polygon = rois[2].contour_points;
        }

        liuzhi_rois_[camera_id] = rois;
        camera_params_[camera_id] = params;
    }

    return !camera_params_.empty();
}

void TedaiJuanquPipeline::updateEnterEvents(int camera_id,
                                            const ReelCameraOutput &out,
                                            const std::vector<ReelDetObject> &raw,
                                            cv::Mat &src, long long now_ms)
{
    if (!config_.events.enable) return;

    // 原始检测框（含中心点在区域外的），供"刚离开区域仍可见"的红框标记使用
    std::vector<ReelRect> raw_boxes;
    raw_boxes.reserve(raw.size());
    for (const auto &o : raw) {
        ReelRect b;
        b.x = o.rect.x;
        b.y = o.rect.y;
        b.width = o.rect.width;
        b.height = o.rect.height;
        b.type = o.classid;
        raw_boxes.push_back(b);
    }

    // 相机 -> 参与事件的区域列表（liuzhi_rois_ 下标 + roi_id）：
    // cam1-3 只判 inside(liuzhi 1)；cam4 同时判 inside(liuzhi 1) 与 outside(liuzhi 2)；
    // cam5 只判 outside(liuzhi 2)；camera0 不参与。每区域每次请求最多 1 条事件。
    std::vector<std::pair<int, const char *>> regions;
    if (camera_id >= 1 && camera_id <= 3) {
        regions.emplace_back(0, "inside");
    } else if (camera_id == 4) {
        regions.emplace_back(0, "inside");
        regions.emplace_back(1, "outside");
    } else if (camera_id == 5) {
        regions.emplace_back(1, "outside");
    } else {
        return;  // camera0 不参与
    }

    auto it = liuzhi_rois_.find(camera_id);
    if (it == liuzhi_rois_.end()) return;

    for (size_t r = 0; r < regions.size(); ++r) {
        int roi_index = regions[r].first;
        const char *roi_id = regions[r].second;
        if ((int)it->second.size() <= roi_index) continue;
        const ReelRoiInfo &roi = it->second[roi_index];

        // 各相机的区域输出均为检测像素坐标，直接用于入口判定与连续性匹配
        const std::vector<ReelRect> &boxes =
            (roi_index == 0) ? out.inside : out.outside;

        ReelEventGate gate;
        if (roi.qishi_zhongzhi_points.size() >= 2) {
            gate.has_axis = true;
            gate.qishi = roi.qishi_zhongzhi_points[0];
            gate.zhongzhi = roi.qishi_zhongzhi_points[1];

            // 出口感应线：过 zhongzhi、垂直于 qishi->zhongzhi 轴，
            // 长度取 ROI 多边形外接矩形对角线（下限 300px），
            // 保证覆盖通道宽度，无需现场额外标定
            if (roi.contour_points.size() >= 3) {
                std::vector<cv::Point> roi_pts;
                roi_pts.reserve(roi.contour_points.size());
                for (const auto &p : roi.contour_points) {
                    roi_pts.emplace_back(p.x, p.y);
                }
                cv::Rect br = cv::boundingRect(roi_pts);
                const double diag =
                    std::sqrt((double)br.width * br.width +
                              (double)br.height * br.height);
                const double half = std::max(diag * 0.5, 150.0);
                const double dx =
                    (double)gate.zhongzhi.x - (double)gate.qishi.x;
                const double dy =
                    (double)gate.zhongzhi.y - (double)gate.qishi.y;
                const double len = std::sqrt(dx * dx + dy * dy);
                if (len > 1e-3) {
                    const double px = -dy / len;
                    const double py = dx / len;
                    gate.gate_a.x =
                        cvRound(gate.zhongzhi.x + px * half);
                    gate.gate_a.y =
                        cvRound(gate.zhongzhi.y + py * half);
                    gate.gate_b.x =
                        cvRound(gate.zhongzhi.x - px * half);
                    gate.gate_b.y =
                        cvRound(gate.zhongzhi.y - py * half);
                    gate.has_gate = true;
                }
            }
        }

        ReelEnterDebug debug;
        enter_tracker_.update(camera_id, roi_id, roi.liuzhi_id, boxes, gate,
                              now_ms, &last_events_, &debug, raw_boxes);

        if (config_.events.draw_debug) {
            // cam4 两个区域各画一份标注，信息面板按序号上移避免重叠
            drawEnterDebug(src, camera_id, debug, config_.events.entry_radius_px,
                           config_.events.entry_t_ratio, (int)r);
        }
    }
}

ReelDetector *TedaiJuanquPipeline::detector(int camera_id)
{
    auto it = detectors_.find(camera_id);
    if (it != detectors_.end()) return it->second.get();

    std::string model_path;
    for (const auto &cam : config_.cameras) {
        if (cam.camera_id == camera_id) {
            model_path = cam.det_model;
            break;
        }
    }
    if (model_path.empty()) {
        std::cerr << "[ERROR] tedai_juanqu: no det model configured for camera "
                  << camera_id << std::endl;
        detectors_[camera_id] = nullptr;
        return nullptr;
    }

    auto det = std::make_unique<ReelDetector>(model_path, config_.device == "cuda",
                                              camera_id);
    if (!det->valid()) {
        std::cerr << "[ERROR] tedai_juanqu: camera " << camera_id
                  << " detector init failed: " << model_path << std::endl;
        detectors_[camera_id] = nullptr;
        return nullptr;
    }
    ReelDetector *raw = det.get();
    detectors_[camera_id] = std::move(det);
    return raw;
}

ReelClassifier *TedaiJuanquPipeline::classifier()
{
    if (classifier_loaded_) return classifier_.get();
    classifier_loaded_ = true;

    if (!classify_enabled_) {
        return nullptr;
    }
    if (config_.classify_model.empty()) {
        std::cerr << "[ERROR] tedai_juanqu: classify_model not configured" << std::endl;
        return nullptr;
    }
    auto cls = std::make_unique<ReelClassifier>(config_.classify_model,
                                                config_.device == "cuda");
    if (!cls->valid()) {
        std::cerr << "[ERROR] tedai_juanqu: classifier init failed: "
                  << config_.classify_model << std::endl;
        return nullptr;
    }
    classifier_ = std::move(cls);
    return classifier_.get();
}

bool TedaiJuanquPipeline::warmup()
{
    bool all_ok = true;

    // 假推理输入：小黑图即可，检测/分类预处理会各自 letterbox/resize 到
    // 模型输入尺寸，推理本身才是预热目标（session、CUDA kernel、显存）。
    cv::Mat dummy(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));

    for (const auto &cam : config_.cameras) {
        ReelDetector *det = detector(cam.camera_id);
        if (det == nullptr) {
            all_ok = false;
            continue;
        }
        std::vector<ReelDetObject> objects;
        if (!det->detect(dummy, objects)) {
            all_ok = false;
        }
    }

    if (classify_enabled_) {
        ReelClassifier *cls = classifier();
        if (cls == nullptr) {
            all_ok = false;
        } else {
            cls->detect(dummy);  // 结果丢弃，仅触发推理初始化
        }
    }

    return all_ok;
}

std::vector<ReelCameraOutput> TedaiJuanquPipeline::process(
    std::vector<ReelFrameInput> &frames)
{
    std::vector<ReelCameraOutput> results;

    // 每请求一次汇总：推理计时清零
    detect_total_ms_ = 0;
    classify_total_ms_ = 0;
    last_timings_.clear();
    last_diagnostics_.clear();
    last_events_.clear();

    const long long event_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // 按 camera_id 升序处理（跨相机去重依赖：camera 5 依赖 camera 3 的本请求结果）
    std::map<int, ReelFrameInput *> frames_by_id;
    for (auto &fr : frames) {
        frames_by_id[fr.camera_id] = &fr;
    }

    std::map<int, ReelCameraOutput> done;  // 本请求内已处理相机结果
    // 保存任务（与 results 同序），返回前统一 get() 保证文件已落盘
    std::vector<std::pair<size_t, std::future<std::string>>> save_futures;

    for (const auto &kv : frames_by_id) {
        int camera_id = kv.first;
        ReelFrameInput *fr = kv.second;

        ReelStepTiming st;
        st.camera_id = camera_id;
        auto t_iter0 = std::chrono::high_resolution_clock::now();

        ReelCameraOutput out;
        out.camera_id = camera_id;

        ReelCameraDiagnostics diag;
        diag.camera_id = camera_id;

        auto params_it = camera_params_.find(camera_id);
        if (fr->image.empty() || params_it == camera_params_.end()) {
            out.read_picture_flag = 0;
            diag.det_ok = false;
        } else {
            // 请求内解码图直接用于绘制（frames 生命周期覆盖本调用，无需整帧 clone）
            cv::Mat &src = fr->image;
            if (camera_id == 0) {
                detectReelExport(camera_id, src, params_it->second, out, st, diag);
            } else {
                const ReelCameraOutput *prev_result = nullptr;
                const ReelRoiParams *prev_params = nullptr;
                if (camera_id == 5) {
                    auto it = done.find(3);
                    auto pit = camera_params_.find(3);
                    if (it != done.end() && pit != camera_params_.end()) {
                        prev_result = &it->second;
                        prev_params = &pit->second;
                    }
                }
                detectReelLocation(camera_id, src, params_it->second,
                                   prev_result, prev_params, out, st, diag);
            }
            out.read_picture_flag = 1;
            // 盘卷进入区域事件（仅检测成功的相机）
            updateEnterEvents(camera_id, out, diag.raw, src, event_now_ms);
            // 保存任务并行执行（编码+写盘），本线程继续处理下一相机
            auto t_save0 = std::chrono::high_resolution_clock::now();
            size_t result_index = results.size();
            save_futures.emplace_back(result_index,
                std::async(std::launch::async, [this, camera_id, src]() {
                    return saveResultImage(camera_id, src);
                }));
            auto t_save1 = std::chrono::high_resolution_clock::now();
            st.save_ms =
                std::chrono::duration<double, std::milli>(t_save1 - t_save0).count();
        }

        auto t_iter1 = std::chrono::high_resolution_clock::now();
        st.total_ms =
            std::chrono::duration<double, std::milli>(t_iter1 - t_iter0).count();

        done[camera_id] = out;
        results.push_back(out);
        last_timings_.push_back(st);
        last_diagnostics_.push_back(diag);
    }

    // 等全部保存完成：响应返回前结果图必须已落盘
    for (auto &kv_save : save_futures) {
        results[kv_save.first].result_pic_path = kv_save.second.get();
    }

    return results;
}

void TedaiJuanquPipeline::detectReelLocation(int camera_id, cv::Mat &src,
                                             const ReelRoiParams &params,
                                             const ReelCameraOutput *prev_result,
                                             const ReelRoiParams *prev_params,
                                              ReelCameraOutput &out, ReelStepTiming &st,
                                              ReelCameraDiagnostics &diag)
{
    // camera 5 分类用干净图：绘制前先裁小 ROI（ROI 取自 camera_liuzhi_roi.json
    // 中 camera 5 的 outside_polygon 外接矩形，替代旧版整帧 src_clone 与写死坐标，
    // 省一次全图拷贝）。仅在分类开关开启时才裁剪。
    cv::Mat cam5_clean_crop;
    if (camera_id == 5 && classify_enabled_) {
        auto t_crop0 = std::chrono::high_resolution_clock::now();
        if (params.outside_polygon.size() >= 3) {
            std::vector<cv::Point> outside_pts;
            outside_pts.reserve(params.outside_polygon.size());
            for (const auto &p : params.outside_polygon) {
                outside_pts.emplace_back(p.x, p.y);
            }
            cv::Rect roi = cv::boundingRect(outside_pts);
            cv::Rect image_bounds(0, 0, src.cols, src.rows);
            cv::Rect safe_roi = roi & image_bounds;
            if (safe_roi.width > 0 && safe_roi.height > 0) {
                cam5_clean_crop = src(safe_roi).clone();
            }
        }
        auto t_crop1 = std::chrono::high_resolution_clock::now();
        st.clone_ms +=
            std::chrono::duration<double, std::milli>(t_crop1 - t_crop0).count();
    }

    std::vector<ReelDetObject> vec_obj;
    ReelDetector *det = detector(camera_id);
    auto t_det0 = std::chrono::high_resolution_clock::now();
    bool det_ok = det && det->detect(src, vec_obj);
    auto t_det1 = std::chrono::high_resolution_clock::now();
    if (!det_ok) {
        vec_obj.clear();
    }
    diag.det_ok = det_ok;
    diag.raw = vec_obj;
    detect_total_ms_ +=
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();
    st.detect_ms =
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();

    auto t_assign0 = std::chrono::high_resolution_clock::now();

    // 三区域多边形（Point 用于绘制，PointD 用于判定）
    std::vector<cv::Point> polygon1, polygon2, polygon3;
    std::vector<ReelPointD> inside_poly, outside_poly, collect_poly;
    for (const auto &p : params.inside_polygon) {
        inside_poly.push_back({(double)p.x, (double)p.y});
        polygon1.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.outside_polygon) {
        outside_poly.push_back({(double)p.x, (double)p.y});
        polygon2.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.collect_polygon) {
        collect_poly.push_back({(double)p.x, (double)p.y});
        polygon3.emplace_back(p.x, p.y);
    }
    if (polygon1.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon1}, true,
                      cv::Scalar(0, 0, 255), 4);
    }
    if (polygon2.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon2}, true,
                      cv::Scalar(0, 255, 255), 4);
    }
    if (polygon3.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon3}, true,
                      cv::Scalar(255, 0, 255), 4);
    }

    // camera 5：下卷 2 号工位分类（ROI 取自 outside_polygon 外接矩形；旧版在每个
    // 检测框循环内重复推理同一张裁剪图，结果等价，这里每帧推理一次）。
    // 仅当 config.enable_classify 开启且分类模型可用时执行。
    int classify_result_id = -1;
    if (camera_id == 5 && classify_enabled_ && !cam5_clean_crop.empty()) {
        ReelClassifier *cls = classifier();
        if (cls && cls->valid()) {
            cv::Mat cropped = cam5_clean_crop;
            auto t_cls0 = std::chrono::high_resolution_clock::now();
            auto cls_result = cls->detect(cropped);
            auto t_cls1 = std::chrono::high_resolution_clock::now();
            classify_total_ms_ +=
                std::chrono::duration<double, std::milli>(t_cls1 - t_cls0).count();
            st.classify_ms =
                std::chrono::duration<double, std::milli>(t_cls1 - t_cls0).count();
            if (cls_result.valid) {
                classify_result_id = cls_result.class_id;
                cv::putText(src, std::to_string(cls_result.class_id),
                            cv::Point(src.cols - 1000, 200),
                            cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(255, 0, 255), 2, 0);
            }
        }
    }

    // 三区域判定（testPoint 为框中心点）
    std::vector<ReelRect> inside_rects, outside_rects, collect_rects;
    for (const auto &obj : vec_obj) {
        ReelRect box;
        box.x = obj.rect.x;
        box.y = obj.rect.y;
        box.width = obj.rect.width;
        box.height = obj.rect.height;
        box.type = obj.classid;

        ReelPointD test_point{(double)obj.rect.x, (double)obj.rect.y};

        if (inside_poly.size() >= 3 && isPointInPolygon(test_point, inside_poly)) {
            if (obj.classid == 0) {
                inside_rects.push_back(box);
            }
        }
        if (outside_poly.size() >= 3 && isPointInPolygon(test_point, outside_poly)) {
            if (camera_id == 4) {
                if (obj.classid == 0 || obj.classid == 1) {
                    outside_rects.push_back(box);
                }
            } else if (camera_id == 5) {
                // 针对误检：正常盘卷 + 面积 > 50000；分类开关开启时额外要求分类结果为 2
                bool keep = (obj.classid == 0 && obj.rect.area() > 50000);
                if (classify_enabled_) {
                    keep = keep && (classify_result_id == 2);
                }
                if (keep) {
                    outside_rects.push_back(box);
                }
            } else {
                outside_rects.push_back(box);
            }
        }
        if (collect_poly.size() >= 3 && isPointInPolygon(test_point, collect_poly)) {
            if (camera_id == 5) {
                if (obj.classid == 2) {
                    collect_rects.push_back(box);
                }
            } else {
                collect_rects.push_back(box);
            }
        }
    }
    auto t_assign1 = std::chrono::high_resolution_clock::now();
    st.post_assign_ms =
        std::chrono::duration<double, std::milli>(t_assign1 - t_assign0).count() -
        st.classify_ms;  // classify 已单独计时，从 assign 块中剔除

    int inside_count = (int)inside_rects.size();
    int outside_count = (int)outside_rects.size();
    int collect_count = (int)collect_rects.size();

    // 诊断快照：区域判定后、排序/去重前
    diag.inside_assigned = inside_rects;
    diag.outside_assigned = outside_rects;
    diag.collect_assigned = collect_rects;
    diag.inside_count_before_dedup = inside_count;

    // 排序（相机专属规则）
    auto t_sort0 = std::chrono::high_resolution_clock::now();
    if (camera_id == 1) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYUp);
    } else if (camera_id == 2) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYDown);
    } else if (camera_id == 3) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpXUp);
    } else if (camera_id == 4) {
        // camera4 liuzhi 1：y 从大到小
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYDown);
    }
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    st.post_sort_ms =
        std::chrono::duration<double, std::milli>(t_sort1 - t_sort0).count();

    // 跨区域重复盘卷过滤（camera5：用 camera3 inside 的像素坐标过滤自己的 outside 首盘）
    auto t_dedup0 = std::chrono::high_resolution_clock::now();
    int guolv_flag_outside = 0;
    if (camera_id == 5 && prev_result && prev_params) {
        diag.dedup_outside.prev_camera_id = 3;
        removeChongfuPanjuanBetweenInsideOutside(
            prev_result->inside, outside_rects, outside_count,
            prev_params->inside_qishi, params.outside_qishi, guolv_flag_outside,
            &diag.dedup_outside);
    }
    auto t_dedup1 = std::chrono::high_resolution_clock::now();
    st.post_dedup_ms =
        std::chrono::duration<double, std::milli>(t_dedup1 - t_dedup0).count();
    diag.inside_count_after_dedup = inside_count;
    diag.guolv_flag_outside = guolv_flag_outside;

    // ---- 绘制（与旧服务一致的样式） ----
    auto t_draw0 = std::chrono::high_resolution_clock::now();
    cv::Scalar color_center(0, 255, 0);
    cv::Scalar color_rec(255, 0, 0);
    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); ++i) {
        const ReelRect &detection = inside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_in_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x + box.width, box.y - 40,
                          text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + box.width, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    if (inside_count > 0) {
        cv::putText(src, std::to_string(inside_count),
                    cv::Point(src.cols - 200, 200),
                    cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);
    }

    cv::Scalar color_center_1(125, 255, 0);
    cv::Scalar color_rec_1(255, 255, 0);
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); ++i) {
        const ReelRect &detection = outside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_1, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_1, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_out_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_1, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    if (outside_count > 0) {
        cv::putText(src, std::to_string(outside_count),
                    cv::Point(src.cols - 200, 200),
                    cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);
    }
    if (guolv_flag_outside == 1) {
        cv::line(src, cv::Point(2000, 0), cv::Point(2000, 500),
                 cv::Scalar(0, 255, 0), 3);
    }

    cv::Scalar color_center_2(125, 255, 0);
    cv::Scalar color_rec_2(255, 0, 255);
    for (int i = 0; i < collect_count && i < (int)collect_rects.size(); ++i) {
        const ReelRect &detection = collect_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_2, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_2, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_collect_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_2, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    auto t_draw1 = std::chrono::high_resolution_clock::now();
    st.post_draw_ms =
        std::chrono::duration<double, std::milli>(t_draw1 - t_draw0).count();

    // 数量截断（旧结构体数组上限 50）
    if (inside_count > kMaxPanjuanCount) inside_count = kMaxPanjuanCount;
    if (outside_count > kMaxPanjuanCount) outside_count = kMaxPanjuanCount;
    if (collect_count > kMaxPanjuanCount) collect_count = kMaxPanjuanCount;

    diag.final_inside = inside_count;
    diag.final_outside = outside_count;
    diag.final_collect = collect_count;

    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); i++) {
        out.inside.push_back(inside_rects[i]);
    }
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); i++) {
        out.outside.push_back(outside_rects[i]);
    }
    for (int i = 0; i < collect_count && i < (int)collect_rects.size(); i++) {
        out.collect.push_back(collect_rects[i]);
    }
}

void TedaiJuanquPipeline::detectReelExport(int camera_id, cv::Mat &src,
                                           const ReelRoiParams &params,
                                           ReelCameraOutput &out, ReelStepTiming &st,
                                           ReelCameraDiagnostics &diag)
{
    // 白色遮挡三个固定区域（旧版硬编码，防止误检盘卷中的钢卷和废钢）
    auto t_clone0 = std::chrono::high_resolution_clock::now();
    cv::Mat src_copy = src.clone();
    auto t_clone1 = std::chrono::high_resolution_clock::now();
    st.clone_ms +=
        std::chrono::duration<double, std::milli>(t_clone1 - t_clone0).count();
    cv::Rect roi1(816, 493, 550, 290);
    cv::Rect roi2(722, 790, 570, 310);
    cv::Rect roi3(600, 280, 310, 200);
    cv::Rect image_bounds(0, 0, src_copy.cols, src_copy.rows);
    auto t_white0 = std::chrono::high_resolution_clock::now();
    if ((roi1 & image_bounds) == roi1) src_copy(roi1) = cv::Scalar(255, 255, 255);
    if ((roi2 & image_bounds) == roi2) src_copy(roi2) = cv::Scalar(255, 255, 255);
    if ((roi3 & image_bounds) == roi3) src_copy(roi3) = cv::Scalar(255, 255, 255);
    auto t_white1 = std::chrono::high_resolution_clock::now();
    st.post_assign_ms +=
        std::chrono::duration<double, std::milli>(t_white1 - t_white0).count();

    std::vector<ReelDetObject> vec_obj;
    ReelDetector *det = detector(camera_id);
    auto t_det0 = std::chrono::high_resolution_clock::now();
    bool det_ok = det && det->detect(src_copy, vec_obj);
    auto t_det1 = std::chrono::high_resolution_clock::now();
    if (!det_ok) {
        vec_obj.clear();
    }
    diag.det_ok = det_ok;
    diag.raw = vec_obj;
    detect_total_ms_ +=
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();
    st.detect_ms =
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();

    auto t_assign0 = std::chrono::high_resolution_clock::now();

    // 正常盘卷过滤：classid==0 且宽>50、高>30；异常盘卷 classid==1（本流程不输出）
    const int set_reel_width_thresh = 50;
    const int set_reel_height_thresh = 30;
    std::vector<ReelDetObject> output_normal;
    for (const auto &obj : vec_obj) {
        if (obj.classid == 0 && obj.rect.width > set_reel_width_thresh &&
            obj.rect.height > set_reel_height_thresh) {
            output_normal.push_back(obj);
        }
    }

    // 两区域多边形
    std::vector<cv::Point> polygon1, polygon2;
    std::vector<ReelPointD> inside_poly, outside_poly;
    for (const auto &p : params.inside_polygon) {
        inside_poly.push_back({(double)p.x, (double)p.y});
        polygon1.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.outside_polygon) {
        outside_poly.push_back({(double)p.x, (double)p.y});
        polygon2.emplace_back(p.x, p.y);
    }
    if (polygon1.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon1}, true,
                      cv::Scalar(0, 0, 255), 4);
    }
    if (polygon2.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon2}, true,
                      cv::Scalar(0, 255, 255), 4);
    }

    std::vector<ReelRect> inside_rects, outside_rects;
    for (const auto &obj : output_normal) {
        ReelRect box;
        box.x = obj.rect.x;
        box.y = obj.rect.y;
        box.width = obj.rect.width;
        box.height = obj.rect.height;
        box.type = obj.classid;

        ReelPointD test_point{(double)obj.rect.x, (double)obj.rect.y};
        if (inside_poly.size() >= 3 && isPointInPolygon(test_point, inside_poly)) {
            if (obj.classid == 0) {
                inside_rects.push_back(box);
            }
        }
        if (outside_poly.size() >= 3 && isPointInPolygon(test_point, outside_poly)) {
            if (obj.classid == 0) {
                outside_rects.push_back(box);
            }
        }
    }
    auto t_assign1 = std::chrono::high_resolution_clock::now();
    st.post_assign_ms +=
        std::chrono::duration<double, std::milli>(t_assign1 - t_assign0).count();

    int inside_count = (int)inside_rects.size();
    int outside_count = (int)outside_rects.size();

    diag.inside_assigned = inside_rects;
    diag.outside_assigned = outside_rects;
    diag.inside_count_before_dedup = inside_count;
    diag.inside_count_after_dedup = inside_count;

    auto t_sort0 = std::chrono::high_resolution_clock::now();
    std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYUp);
    std::sort(outside_rects.begin(), outside_rects.end(), reelRectCmpYUp);
    auto t_sort1 = std::chrono::high_resolution_clock::now();
    st.post_sort_ms =
        std::chrono::duration<double, std::milli>(t_sort1 - t_sort0).count();

    // 绘制（旧版样式：inside 文本在框右上）
    auto t_draw0 = std::chrono::high_resolution_clock::now();
    cv::Scalar color_center(0, 255, 0);
    cv::Scalar color_rec(255, 0, 0);
    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); ++i) {
        const ReelRect &detection = inside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_in_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    cv::putText(src, std::to_string(inside_count), cv::Point(src.cols - 200, 200),
                cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);

    cv::Scalar color_center_1(125, 255, 0);
    cv::Scalar color_rec_1(255, 255, 0);
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); ++i) {
        const ReelRect &detection = outside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_1, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_1, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_out_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_1, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    auto t_draw1 = std::chrono::high_resolution_clock::now();
    st.post_draw_ms =
        std::chrono::duration<double, std::milli>(t_draw1 - t_draw0).count();

    if (inside_count > kMaxPanjuanCount) inside_count = kMaxPanjuanCount;
    if (outside_count > kMaxPanjuanCount) outside_count = kMaxPanjuanCount;

    diag.final_inside = inside_count;
    diag.final_outside = outside_count;

    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); i++) {
        out.inside.push_back(inside_rects[i]);
    }
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); i++) {
        out.outside.push_back(outside_rects[i]);
    }
}

std::string TedaiJuanquPipeline::saveResultImage(int camera_id, const cv::Mat &image)
{
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm lt_tm{};
    // 保存任务多线程并行，localtime 非线程安全
#ifdef _WIN32
    localtime_s(&lt_tm, &t);
#else
    localtime_r(&t, &lt_tm);
#endif
    std::tm *lt = &lt_tm;

    std::string folder =
        cv::format("%s\\camera_%02d", config_.result_dir.c_str(), camera_id);
    FileHelper::ensureDirectoryExists(folder);
    std::string date_folder = cv::format("%s\\%d%02d%02d", folder.c_str(),
                                         lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday);
    FileHelper::ensureDirectoryExists(date_folder);

    std::string save_name = cv::format(
        "%s\\%d%02d%02d%02d%02d%02d%03d_%02d.jpg", date_folder.c_str(),
        lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
        lt->tm_hour, lt->tm_min, lt->tm_sec, (int)ms.count(), camera_id);

    if (!image.empty()) {
        // 压测开关：JHDEEP_NO_SAVE=1 跳过结果图写盘（测纯算法/推理耗时）
        const char *no_save = getenv("JHDEEP_NO_SAVE");
        if (no_save == nullptr || std::string(no_save) == "0") {
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(75);
            cv::imwrite(save_name, image, compression_params);
        }
    }
    return save_name;
}

} // namespace Pipeline
} // namespace JHDeepCore
