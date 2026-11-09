#include "reel_track_pipeline.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

namespace JHDeepCore {
namespace Pipeline {

namespace {

// 旧版 cmp_y_up / cmp_y_down（Detection 版）：按中心 y + 高/2（下边缘）比较
bool detCmpYUp(const ReelDetObject &a, const ReelDetObject &b)
{
    return (a.rect.y + a.rect.height / 2) < (b.rect.y + b.rect.height / 2);
}

bool detCmpYDown(const ReelDetObject &a, const ReelDetObject &b)
{
    return (a.rect.y + a.rect.height / 2) > (b.rect.y + b.rect.height / 2);
}

// 旧版 reelboxmatch_cmp_y_up：按中心 y 升序
bool boxMatchCmpYUp(const reelBoxMatch &a, const reelBoxMatch &b)
{
    return a.y < b.y;
}

// 当前本地日期 YYYYMMDD。DLL 层已用 g_mutex 串行化 detect，此处线程安全可放宽。
std::string CurrentDateYYYYMMDD()
{
    std::time_t t = std::time(nullptr);
    std::tm tmv = {};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tmv);
    return std::string(buf);
}

// 盘卷标签：深色底（蓝/红）+ 白字。工位用 1.0 字号，完整 ID 用 0.7 字号接在同一行；
// 默认画在框上方边缘，靠近画面顶部时挪到框内，靠右时左移，避免出界。
void DrawReelLabel(cv::Mat &img, const cv::Rect &box,
                   const std::string &station_text, const std::string &reel_id,
                   const cv::Scalar &bg_color)
{
    const int kFont = cv::FONT_HERSHEY_DUPLEX;
    const double kStationScale = 1.0;
    const double kIdScale = 0.7;
    const int kThickness = 2;
    const int kPad = 8;
    const int kGap = 10;
    const cv::Scalar kTextColor(255, 255, 255);  // 白字

    if (station_text.empty() && reel_id.empty()) return;

    int station_base = 0;
    int id_base = 0;
    cv::Size station_size =
        station_text.empty()
            ? cv::Size(0, 0)
            : cv::getTextSize(station_text, kFont, kStationScale, kThickness,
                              &station_base);
    cv::Size id_size =
        reel_id.empty() ? cv::Size(0, 0)
                        : cv::getTextSize(reel_id, kFont, kIdScale, kThickness,
                                          &id_base);

    int text_h = std::max(station_size.height, id_size.height);
    int label_w = station_size.width + kPad * 2;
    if (!reel_id.empty()) {
        label_w += (station_text.empty() ? 0 : kGap) + id_size.width;
    }
    int label_h = text_h + kPad * 2;

    int x = box.x;
    int y = box.y - label_h - 2;  // 框上方
    if (y < 0) y = box.y + 2;     // 顶部越界：画到框内
    if (x + label_w > img.cols) x = img.cols - label_w;  // 右侧越界：左移
    if (x < 0) x = 0;

    cv::rectangle(img, cv::Rect(x, y, label_w, label_h), bg_color, cv::FILLED);

    // 两段文字共用基线，底部对齐
    int baseline_y = y + kPad + text_h;
    if (!station_text.empty()) {
        cv::putText(img, station_text, cv::Point(x + kPad, baseline_y), kFont,
                    kStationScale, kTextColor, kThickness, cv::LINE_AA);
    }
    if (!reel_id.empty()) {
        int id_x = x + kPad + station_size.width +
                   (station_text.empty() ? 0 : kGap);
        cv::putText(img, reel_id, cv::Point(id_x, baseline_y), kFont, kIdScale,
                    kTextColor, kThickness, cv::LINE_AA);
    }
}

// 拼接绘制摘要（标签内容变化时 DLL 写日志，方便现场定位"没有号"的问题）。
void AppendDrawLabel(std::string &out, const std::string &label)
{
    if (!out.empty()) out += "|";
    out += label;
}

} // namespace

bool ReelTrackPipeline::init(const std::string &model_path, bool use_gpu,
                             ReelIdCallback on_reel_id_assigned)
{
    detector_ = std::make_unique<ReelDetector>(model_path, use_gpu);
    if (!detector_->valid()) {
        std::cerr << "[ERROR] reel track detector init failed: " << model_path
                  << std::endl;
        detector_.reset();
        return false;
    }
    // 报警跟踪状态复位（旧版 static 状态在进程内不清除；实例重建即复位）
    prev_station_one_real_state_ = 0;
    prev_station_two_real_state_ = 0;
    prev_reels_.clear();
    // 赋号状态复位；持久化序号由 DLL 层通过 seedReelOrder 恢复
    reel_order_counter_ = 0;
    reel_id_date_.clear();
    last_reel_id_.clear();
    last_reel_id_value_ = 0;
    reel_ids_.clear();
    reported_leave_ids_.clear();
    leave_events_.clear();
    birth_events_.clear();
    last_draw_summary_.clear();
    on_reel_id_assigned_ = std::move(on_reel_id_assigned);
    return true;
}

void ReelTrackPipeline::seedReelOrder(const std::string &date, int seq)
{
    if (date.empty() || seq <= 0) return;
    if (date != CurrentDateYYYYMMDD()) return;  // 非当天不恢复，交给跨天重置
    reel_id_date_ = date;
    reel_order_counter_ = seq;
}

// 出生赋号：当天自增序号（跨天重置），写入 box.order_number，并记录完整唯一值。
// ID 为纯数字 YYYYMMDDNNN（例：20261104007），方便 C# 用 long 保存。
void ReelTrackPipeline::assignNewReelOrder(reelBoxMatch &box)
{
    std::string today = CurrentDateYYYYMMDD();
    if (today != reel_id_date_) {
        reel_id_date_ = today;
        reel_order_counter_ = 0;
        reel_ids_.clear();            // 只保留当天号码用于图上标注
        reported_leave_ids_.clear();  // 跨天清空离开事件去重记录
    }
    ++reel_order_counter_;
    box.order_number = reel_order_counter_;
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%03d", today.c_str(), reel_order_counter_);
    last_reel_id_ = buf;
    last_reel_id_value_ = std::strtoll(buf, nullptr, 10);
    reel_ids_.push_back(last_reel_id_);
    // 出生事件入队：号 + 序号 + 工位 + 出生帧坐标，供 DLL 层转成 C ABI 事件。
    // 调用点都已填好 box 的 x/y/width/height/station_number。
    {
        ReelBirthEventInfo info;
        info.reel_id = last_reel_id_value_;
        info.order_number = box.order_number;
        info.station_number = box.station_number;
        info.x = box.x;
        info.y = box.y;
        info.width = box.width;
        info.height = box.height;
        birth_events_.push_back(info);
    }
    if (on_reel_id_assigned_) {
        on_reel_id_assigned_(reel_id_date_, reel_order_counter_);
    }
}

std::string ReelTrackPipeline::reelIdText(int order) const
{
    if (order <= 0 || order > reel_order_counter_) return std::string();
    size_t idx = (size_t)(order - 1);
    if (idx < reel_ids_.size() && !reel_ids_[idx].empty()) {
        return reel_ids_[idx];
    }
    // 跨进程重启后，重启前出生的卷不在表内：按当前日期 + 序号兜底
    if (reel_id_date_.empty()) return std::string();
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%03d", reel_id_date_.c_str(), order);
    return std::string(buf);
}

long long ReelTrackPipeline::reelIdValue(int order) const
{
    std::string text = reelIdText(order);
    if (text.empty()) return 0;
    return std::strtoll(text.c_str(), nullptr, 10);
}

std::vector<ReelLeaveEventInfo> ReelTrackPipeline::takeLeaveEvents()
{
    std::vector<ReelLeaveEventInfo> events;
    events.swap(leave_events_);
    return events;
}

std::vector<ReelBirthEventInfo> ReelTrackPipeline::takeBirthEvents()
{
    std::vector<ReelBirthEventInfo> events;
    events.swap(birth_events_);
    return events;
}

bool ReelTrackPipeline::detect(cv::Mat &src_image,
                               const InputParamsReelExportTargetTrack &input_params,
                               OutputResultReelExportTargetTrack *output_params)
{
    if (!valid()) return false;

    // ========== 数组越界保护（旧版：输入数量超出 [0,10] 截断为 0） ==========
    int exist_prev_count = input_params.exist_region_previous_frame_boxCount;
    int leave_prev_count = input_params.leave_region_previous_frame_boxCount;
    if (exist_prev_count < 0 || exist_prev_count > 10) exist_prev_count = 0;
    if (leave_prev_count < 0 || leave_prev_count > 10) leave_prev_count = 0;

    // ========== 输入参数 ==========
    const int set_station_one_fengang_line0 = input_params.set_station_one_fengang_line;
    const int set_station_two_fengang_line0 = input_params.set_station_two_fengang_line;
    const int set_leave_fengang_line0 = input_params.set_leave_fengang_line;
    const int previous_frame_leave_near_fengang_line_right0 =
        input_params.previous_frame_leave_near_fengang_line_right;
    const int previous_frame_leave_station_number0 =
        input_params.previous_frame_leave_station_number;
    const int set_detect_deviation_pixel_value0 =
        input_params.set_detect_deviation_pixel_value;
    const int set_reel_width_thresh0 = input_params.set_reel_width_thresh;
    const int set_reel_height_thresh0 = input_params.set_reel_height_thresh;

    // ========== 白色遮挡三个固定区域（防止检测到正在盘卷的钢卷/废钢） ==========
    cv::Mat src_copy = src_image.clone();
    cv::Rect roi1(816, 493, 550, 290);
    cv::Rect roi2(722, 790, 570, 310);
    cv::Rect roi3(600, 280, 310, 200);
    if (src_image.cols < 1366 || src_image.rows < 1100) {
        // 旧行为：小图跳过 ROI 遮挡（静默退化，仅日志）
        std::cout << "[WARN] reel track: image (" << src_image.cols << "x"
                  << src_image.rows
                  << ") smaller than ROI area (1366x1100), skip occlusion"
                  << std::endl;
    } else {
        cv::Scalar fixed_value(255, 255, 255);
        src_copy(roi1) = fixed_value;
        src_copy(roi2) = fixed_value;
        src_copy(roi3) = fixed_value;
    }

    // ========== 检测推理 ==========
    std::vector<ReelDetObject> vec_obj;
    if (!detector_->detect(src_copy, vec_obj)) {
        vec_obj.clear();
    }

    // ========== 分类：正常 / 异常（小卷）/ 其他 ==========
    std::vector<ReelDetObject> output_normal;    // 正常盘卷
    std::vector<ReelDetObject> output_abnormal;  // 异常盘卷（没卷完的小卷）
    std::vector<ReelDetObject> output_other;     // 其他情况
    for (const auto &obj : vec_obj) {
        if (obj.classid < 0 || obj.classid >= 2) continue;  // labels {panjuan, abnormal}
        if (obj.classid == 0 && obj.rect.width > set_reel_width_thresh0 &&
            obj.rect.height > set_reel_height_thresh0) {
            output_normal.push_back(obj);
        } else if (obj.classid == 1) {
            output_abnormal.push_back(obj);
        } else {
            output_other.push_back(obj);
        }
    }

    // ========== 画三条预设分钢线 ==========
    cv::Scalar color_line0(0, 255, 255);
    cv::line(src_image, cv::Point(100, set_station_one_fengang_line0),
             cv::Point(800, set_station_one_fengang_line0), color_line0, 2);
    cv::line(src_image, cv::Point(100, set_station_two_fengang_line0),
             cv::Point(800, set_station_two_fengang_line0), color_line0, 2);
    cv::line(src_image, cv::Point(100, set_leave_fengang_line0),
             cv::Point(800, set_leave_fengang_line0), color_line0, 2);

    // ========== 上一帧盘卷坐标（按 y 升序） ==========
    std::vector<reelBoxMatch> pre_frame_exist;
    for (int i = 0; i < exist_prev_count; i++) {
        pre_frame_exist.push_back(input_params.boxs_exist_region_previous[i]);
    }
    std::sort(pre_frame_exist.begin(), pre_frame_exist.end(), boxMatchCmpYUp);

    std::vector<reelBoxMatch> pre_frame_leave;
    for (int i = 0; i < leave_prev_count; i++) {
        pre_frame_leave.push_back(input_params.boxs_leave_region_previous[i]);
    }
    std::sort(pre_frame_leave.begin(), pre_frame_leave.end(), boxMatchCmpYUp);

    // ========== 输出变量初始化 ==========
    int exist_region_current_frame_boxCount0 = 0;  // 有效区域内（分钢线左侧）
    int leave_region_current_frame_boxCount0 = 0;  // 离开区域（分钢线右侧）
    int current_frame_leave_near_fengang_line_right0 = 5000;
    int current_frame_leave_station_number0 = 0;
    int station_one_panjuan_out_state0 = 0;
    int station_two_panjuan_out_state0 = 0;
    int panjuan_back_state0 = 0;

    // ========== 当前帧左右区域划分（按离开分钢线） ==========
    std::vector<ReelDetObject> left_points;   // 左区域（y 小），按 y 降序
    std::vector<ReelDetObject> right_points;  // 右区域（y 大），按 y 升序
    for (const auto &obj : output_normal) {
        if (obj.rect.y < set_leave_fengang_line0) {
            left_points.push_back(obj);
        } else {
            right_points.push_back(obj);
        }
    }
    std::sort(left_points.begin(), left_points.end(), detCmpYDown);
    std::sort(right_points.begin(), right_points.end(), detCmpYUp);

    std::vector<reelBoxMatch> boxs_exist_current;  // 有效区域内的盘卷
    std::vector<reelBoxMatch> boxs_leave_current;  // 离开区域内的盘卷

    // ========== 逻辑处理 ==========
    if (exist_prev_count == 0) {  // 上一帧有效区域内无盘卷
        if (left_points.empty()) {
            // 上一帧 0 当前帧 0：没有盘卷从打卷机出来
            station_one_panjuan_out_state0 = 0;
            station_two_panjuan_out_state0 = 0;
            exist_region_current_frame_boxCount0 = 0;
            current_frame_leave_station_number0 = previous_frame_leave_station_number0;
        } else if (left_points.size() == 1) {
            // 上一帧 0 当前帧 1：按位置判定出处
            if (left_points[0].rect.y < set_station_one_fengang_line0) {
                // 1 号工位分钢线上部 -> 从 1 号出
                station_one_panjuan_out_state0 = 1;
                station_two_panjuan_out_state0 = 0;
                exist_region_current_frame_boxCount0 = 1;
                reelBoxMatch exist_box;
                exist_box.x = left_points[0].rect.x;
                exist_box.y = left_points[0].rect.y;
                exist_box.width = left_points[0].rect.width;
                exist_box.height = left_points[0].rect.height;
                exist_box.station_number = 1;
                assignNewReelOrder(exist_box);
                exist_box.previous_x = left_points[0].rect.x;
                exist_box.previous_y = left_points[0].rect.y;
                exist_box.previous_width = left_points[0].rect.width;
                exist_box.previous_height = left_points[0].rect.height;
                boxs_exist_current.push_back(exist_box);
                current_frame_leave_station_number0 = previous_frame_leave_station_number0;
            } else if (left_points[0].rect.y >= set_station_one_fengang_line0 &&
                       left_points[0].rect.y < set_station_two_fengang_line0) {
                // 1、2 号工位分钢线之间 -> 从 2 号出
                station_one_panjuan_out_state0 = 0;
                station_two_panjuan_out_state0 = 1;
                exist_region_current_frame_boxCount0 = 1;
                reelBoxMatch exist_box;
                exist_box.x = left_points[0].rect.x;
                exist_box.y = left_points[0].rect.y;
                exist_box.width = left_points[0].rect.width;
                exist_box.height = left_points[0].rect.height;
                exist_box.station_number = 2;
                assignNewReelOrder(exist_box);
                exist_box.previous_x = left_points[0].rect.x;
                exist_box.previous_y = left_points[0].rect.y;
                exist_box.previous_width = left_points[0].rect.width;
                exist_box.previous_height = left_points[0].rect.height;
                boxs_exist_current.push_back(exist_box);
                current_frame_leave_station_number0 = previous_frame_leave_station_number0;
            } else {
                // 从分钢线右侧回退
                panjuan_back_state0 = 1;
                station_one_panjuan_out_state0 = 0;
                station_two_panjuan_out_state0 = 0;
                exist_region_current_frame_boxCount0 = 1;
                reelBoxMatch exist_box;
                exist_box.x = left_points[0].rect.x;
                exist_box.y = left_points[0].rect.y;
                exist_box.width = left_points[0].rect.width;
                exist_box.height = left_points[0].rect.height;
                exist_box.station_number = previous_frame_leave_station_number0;
                exist_box.order_number = 0;
                exist_box.previous_x = left_points[0].rect.x;
                exist_box.previous_y = left_points[0].rect.y;
                exist_box.previous_width = left_points[0].rect.width;
                exist_box.previous_height = left_points[0].rect.height;
                boxs_exist_current.push_back(exist_box);
                current_frame_leave_station_number0 = 0;
            }
        } else {
            // 上一帧 0 当前帧 >=2：逐盘按位置判定
            current_frame_leave_station_number0 = previous_frame_leave_station_number0;
            exist_region_current_frame_boxCount0 = (int)left_points.size();
            for (const auto &p : left_points) {
                reelBoxMatch exist_box;
                exist_box.x = p.rect.x;
                exist_box.y = p.rect.y;
                exist_box.width = p.rect.width;
                exist_box.height = p.rect.height;
                exist_box.previous_x = p.rect.x;
                exist_box.previous_y = p.rect.y;
                exist_box.previous_width = p.rect.width;
                exist_box.previous_height = p.rect.height;
                if (p.rect.y < set_station_one_fengang_line0) {
                    station_one_panjuan_out_state0 = 1;
                    exist_box.station_number = 1;
                    assignNewReelOrder(exist_box);
                } else if (p.rect.y >= set_station_one_fengang_line0 &&
                           p.rect.y < set_station_two_fengang_line0) {
                    station_two_panjuan_out_state0 = 1;
                    exist_box.station_number = 2;
                    assignNewReelOrder(exist_box);
                } else {
                    panjuan_back_state0 = 1;
                    exist_box.station_number = previous_frame_leave_station_number0;
                    exist_box.order_number = 0;
                    current_frame_leave_station_number0 = 0;
                }
                boxs_exist_current.push_back(exist_box);
            }
        }
    } else {  // 上一帧有效区域内有盘卷
        if (left_points.empty()) {
            // 上一帧有 当前帧 0：正常离开或目标消失
            station_one_panjuan_out_state0 = 0;
            station_two_panjuan_out_state0 = 0;
            exist_region_current_frame_boxCount0 = 0;
            current_frame_leave_station_number0 =
                pre_frame_exist[0].station_number;
        } else {
            // ========== 最近邻匹配（y 方向 + 线性运动估计） ==========
            std::vector<ReelDetObject> match_success_cur;
            std::vector<reelBoxMatch> match_success_prev;
            int count_number_sum = (int)left_points.size();
            for (int i = 0; i < count_number_sum; i++) {
                int min_distance = INT_MAX;
                int best_cur = -1;
                int best_prev = -1;
                for (int j = 0; j < (int)left_points.size(); j++) {
                    for (int k = 0; k < (int)pre_frame_exist.size(); k++) {
                        // 上一帧运动距离的线性估计（y 方向）
                        int sport_dis =
                            (pre_frame_exist[k].y - pre_frame_exist[k].previous_y) / 2;
                        int pre_estimate_y = pre_frame_exist[k].y + sport_dis;
                        int cur_distance =
                            std::abs(left_points[j].rect.y - pre_estimate_y);
                        if (cur_distance < min_distance) {
                            min_distance = cur_distance;
                            best_cur = j;
                            best_prev = k;
                        }
                    }
                }
                // 像素距离阈值 180：匹配成功
                if (min_distance < 180 && best_cur != -1 && best_prev != -1) {
                    match_success_cur.push_back(left_points[best_cur]);
                    match_success_prev.push_back(pre_frame_exist[best_prev]);
                    left_points.erase(left_points.begin() + best_cur);
                    pre_frame_exist.erase(pre_frame_exist.begin() + best_prev);
                }
            }

            // 匹配失败的当前帧盘卷：按位置判定出处
            int back_flag = 0;
            for (const auto &p : left_points) {
                reelBoxMatch exist_box;
                exist_box.x = p.rect.x;
                exist_box.y = p.rect.y;
                exist_box.width = p.rect.width;
                exist_box.height = p.rect.height;
                exist_box.previous_x = p.rect.x;
                exist_box.previous_y = p.rect.y;
                exist_box.previous_width = p.rect.width;
                exist_box.previous_height = p.rect.height;
                if (p.rect.y < set_station_one_fengang_line0) {
                    station_one_panjuan_out_state0 = 1;
                    exist_box.station_number = 1;
                    assignNewReelOrder(exist_box);
                } else if (p.rect.y >= set_station_one_fengang_line0 &&
                           p.rect.y < set_station_two_fengang_line0) {
                    station_two_panjuan_out_state0 = 1;
                    exist_box.station_number = 2;
                    assignNewReelOrder(exist_box);
                } else {
                    // 从分钢线右侧回退
                    panjuan_back_state0 = 1;
                    back_flag = 1;
                    exist_box.station_number = previous_frame_leave_station_number0;
                    exist_box.order_number = 0;
                    current_frame_leave_station_number0 = 0;
                }
                boxs_exist_current.push_back(exist_box);
            }
            (void)back_flag;  // 旧版仅用于日志判定，无后续逻辑

            // 匹配成功的盘卷：继承工位号，记录上一帧坐标
            for (size_t i = 0; i < match_success_cur.size(); i++) {
                reelBoxMatch exist_box;
                exist_box.x = match_success_cur[i].rect.x;
                exist_box.y = match_success_cur[i].rect.y;
                exist_box.width = match_success_cur[i].rect.width;
                exist_box.height = match_success_cur[i].rect.height;
                exist_box.station_number = match_success_prev[i].station_number;
                exist_box.order_number = match_success_prev[i].order_number;
                exist_box.previous_x = match_success_prev[i].x;
                exist_box.previous_y = match_success_prev[i].y;
                exist_box.previous_width = match_success_prev[i].width;
                exist_box.previous_height = match_success_prev[i].height;
                boxs_exist_current.push_back(exist_box);
            }
            exist_region_current_frame_boxCount0 = (int)boxs_exist_current.size();
        }
    }

    // ========== 离开区域（分钢线右侧）盘卷更新 ==========
    leave_region_current_frame_boxCount0 = (int)right_points.size();
    for (const auto &p : right_points) {
        reelBoxMatch leave_box{};  // 值初始化：避免 order_number 等字段带脏值
        leave_box.x = p.rect.x;
        leave_box.y = p.rect.y;
        leave_box.width = p.rect.width;
        leave_box.height = p.rect.height;
        boxs_leave_current.push_back(leave_box);
    }
    // 离开区域没有自己的跟踪状态：用上一帧 exist + leave 做位置预测匹配，
    // 一对一继承 order_number / station_number，供图上标注与离开事件使用。
    if (!boxs_leave_current.empty()) {
        std::vector<reelBoxMatch> prev_candidates;
        for (int i = 0; i < exist_prev_count; i++) {
            prev_candidates.push_back(input_params.boxs_exist_region_previous[i]);
        }
        for (int i = 0; i < leave_prev_count; i++) {
            prev_candidates.push_back(input_params.boxs_leave_region_previous[i]);
        }
        std::vector<bool> candidate_used(prev_candidates.size(), false);
        for (auto &leave_box : boxs_leave_current) {
            int best_index = -1;
            int best_distance = INT_MAX;
            for (size_t k = 0; k < prev_candidates.size(); k++) {
                if (candidate_used[k]) continue;
                const reelBoxMatch &prev = prev_candidates[k];
                if (prev.order_number <= 0) continue;
                // 横向偏差过大视为不同的卷（防止相邻盘卷串号）
                if (std::abs(leave_box.x - prev.x) > 120) continue;
                // 用上一帧位移做线性外推，盘卷移动快时也能接上号
                int predict_y = prev.y;
                if (prev.previous_y > 0) {
                    predict_y = prev.y + (prev.y - prev.previous_y) / 2;
                }
                int distance = std::abs(leave_box.y - predict_y);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_index = (int)k;
                }
            }
            if (best_index >= 0 && best_distance < 180) {
                const reelBoxMatch &prev = prev_candidates[best_index];
                leave_box.order_number = prev.order_number;
                leave_box.station_number = prev.station_number;
                leave_box.previous_x = prev.x;
                leave_box.previous_y = prev.y;
                leave_box.previous_width = prev.width;
                leave_box.previous_height = prev.height;
                candidate_used[best_index] = true;
            }
        }
    }
    if (right_points.empty()) {
        current_frame_leave_near_fengang_line_right0 = 5000;
    } else {
        // 右侧最靠近分钢线的盘卷（right_points 已按 y 升序）
        current_frame_leave_near_fengang_line_right0 =
            right_points[0].rect.y + right_points[0].rect.height / 2;
    }

    // 注：旧版 change_state 分钢线附近跳变消抖计算结果从未被使用（死代码），不迁移

    // ========== 绘制：有效区域内的盘卷（蓝框 + 工位标签） ==========
    cv::Scalar color_center(0, 255, 0);
    cv::Scalar color_rec(255, 0, 0);
    std::string draw_labels;
    int draw_with_id = 0;
    for (int i = 0; i < exist_region_current_frame_boxCount0 &&
                    i < (int)boxs_exist_current.size();
         ++i) {
        const reelBoxMatch &detection = boxs_exist_current[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src_image, box, color_rec, 2);
        cv::circle(src_image, cv::Point(detection.x, detection.y), 5,
                   color_center, -1);
        std::string class_string;
        if (detection.station_number == 1) {
            class_string = "station1";
        } else if (detection.station_number == 2) {
            class_string = "station2";
        } else {
            class_string = "notknow";
        }
        std::string reel_id = reelIdText(detection.order_number);
        DrawReelLabel(src_image, box, class_string, reel_id, color_rec);
        std::string label = class_string;
        if (!reel_id.empty()) {
            label += " " + reel_id;
            ++draw_with_id;
        }
        AppendDrawLabel(draw_labels, label);
    }

    // ========== 绘制：离开区域盘卷（红框；首个按离开工位标注） ==========
    std::sort(boxs_leave_current.begin(), boxs_leave_current.end(), boxMatchCmpYUp);
    cv::Scalar color_center1(0, 0, 255);
    cv::Scalar color_rec1(0, 0, 255);
    for (int i = 0; i < leave_region_current_frame_boxCount0 &&
                    i < (int)boxs_leave_current.size();
         ++i) {
        const reelBoxMatch &detection = boxs_leave_current[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src_image, box, color_rec1, 2);
        cv::circle(src_image, cv::Point(detection.x, detection.y), 5,
                   color_center1, -1);
        std::string class_string;
        if (i == 0) {
            // 优先用 DLL 记录的离开工位；未知时回退到上一帧跟踪到的工位
            int station = current_frame_leave_station_number0;
            if (station != 1 && station != 2) station = detection.station_number;
            if (station == 1) {
                class_string = "station1";
                boxs_leave_current[i].station_number = 1;
            } else if (station == 2) {
                class_string = "station2";
                boxs_leave_current[i].station_number = 2;
            } else {
                class_string = "notknow";
                boxs_leave_current[i].station_number = -1;
            }
        } else {
            class_string = "notknow";
        }
        std::string reel_id = reelIdText(detection.order_number);
        // 有盘卷号码时不再显示 "notknow"（号码本身已能标识该盘卷）
        if (!reel_id.empty() && class_string == "notknow") class_string.clear();
        DrawReelLabel(src_image, box, class_string, reel_id, color_rec1);
        std::string label = class_string;
        if (!reel_id.empty()) {
            label += " " + reel_id;
            ++draw_with_id;
        }
        AppendDrawLabel(draw_labels, label);
    }

    // 绘制摘要：标签内容变化时由 DLL 写日志（定位"图上没有号"的问题）
    last_draw_summary_ =
        "exist=" + std::to_string(exist_region_current_frame_boxCount0) +
        " leave=" + std::to_string(leave_region_current_frame_boxCount0) +
        " with_id=" + std::to_string(draw_with_id) +
        " labels=" + (draw_labels.empty() ? std::string("<none>") : draw_labels);

    // ========== 盘卷离开事件（首次进入离开区且已带号，同一号码只报一次） ==========
    for (int i = 0; i < leave_region_current_frame_boxCount0 &&
                    i < (int)boxs_leave_current.size();
         ++i) {
        const reelBoxMatch &leave_box = boxs_leave_current[i];
        if (leave_box.order_number <= 0) continue;
        long long reel_id_value = reelIdValue(leave_box.order_number);
        if (reel_id_value <= 0) continue;
        if (reported_leave_ids_.count(reel_id_value) > 0) continue;
        reported_leave_ids_.insert(reel_id_value);
        ReelLeaveEventInfo info;
        info.reel_id = reel_id_value;
        info.station_number = leave_box.station_number;
        info.x = leave_box.x;
        info.y = leave_box.y;
        info.width = leave_box.width;
        info.height = leave_box.height;
        leave_events_.push_back(info);
    }

    // ========== 绘制：异常盘卷（红框） ==========
    cv::Scalar color_center2(0, 0, 255);
    cv::Scalar color_rec2(0, 0, 255);
    for (const auto &obj : output_abnormal) {
        cv::Rect box;
        box.x = obj.rect.x - obj.rect.width / 2;
        box.y = obj.rect.y - obj.rect.height / 2;
        box.height = obj.rect.height;
        box.width = obj.rect.width;
        cv::rectangle(src_image, box, color_rec2, 2);
        cv::circle(src_image, cv::Point(obj.rect.x, obj.rect.y), 5,
                   color_center2, -1);
        std::string class_string = "abnormal";
        DrawReelLabel(src_image, box, class_string, std::string(), color_rec2);
    }

    // ========== 工位出入状态（报警边缘触发前的基础状态） ==========
    output_params->station_one_panjuan_out_state = station_one_panjuan_out_state0;
    output_params->station_two_panjuan_out_state = station_two_panjuan_out_state0;

    // ========== 异常卷边缘触发报警（跨帧状态为实例成员） ==========
    int current_real_state_one = 0;  // 0 无异常，-1 有异常
    int current_real_state_two = 0;
    output_params->station_one_alarm_trigger = 0;
    output_params->station_two_alarm_trigger = 0;

    std::vector<std::pair<int, int>> current_reels;  // (中心x, 出生工位)
    for (const auto &obj : output_abnormal) {
        int center_y = obj.rect.y;
        int center_x = obj.rect.x;
        int matched_source = 0;  // 0 表示未匹配到历史来源

        // 两帧间异常卷移动的最大像素距离容差
        int max_allowed_movement = 40;
        for (const auto &prev_reel : prev_reels_) {
            int dist = std::abs(prev_reel.first - center_x);
            if (dist < max_allowed_movement) {
                max_allowed_movement = dist;
                matched_source = prev_reel.second;  // 继承出生工位
            }
        }

        // 新出现的卷：按 y 判定出生工位（<700 上工位=2，>=700 下工位=1）
        if (matched_source == 0) {
            if (center_y < 700) {
                matched_source = 2;
            } else {
                matched_source = 1;
            }
        }
        current_reels.push_back({center_x, matched_source});

        // 按"出生工位"触发状态，而非当前坐标（防误报）
        if (matched_source == 2) {
            current_real_state_one = -1;
        } else if (matched_source == 1) {
            current_real_state_two = -1;
        }
    }
    prev_reels_ = current_reels;

    if (current_real_state_one == -1) {
        output_params->station_one_panjuan_out_state = -1;
        // 边缘检测：上一帧非 -1 时触发（仅一帧）
        if (prev_station_one_real_state_ != -1) {
            output_params->station_one_alarm_trigger = 1;
        }
    }
    if (current_real_state_two == -1) {
        output_params->station_two_panjuan_out_state = -1;
        if (prev_station_two_real_state_ != -1) {
            output_params->station_two_alarm_trigger = 1;
        }
    }
    prev_station_one_real_state_ = current_real_state_one;
    prev_station_two_real_state_ = current_real_state_two;

    // ========== 输出数量越界保护（超出 [0,10] 截断为 10） ==========
    if (exist_region_current_frame_boxCount0 < 0 ||
        exist_region_current_frame_boxCount0 > 10) {
        exist_region_current_frame_boxCount0 = 10;
    }
    if (leave_region_current_frame_boxCount0 < 0 ||
        leave_region_current_frame_boxCount0 > 10) {
        leave_region_current_frame_boxCount0 = 10;
    }

    // ========== 输出写回 ==========
    output_params->height = src_image.rows;
    output_params->width = src_image.cols;
    output_params->channels = src_image.channels();
    output_params->exist_region_current_frame_boxCount =
        exist_region_current_frame_boxCount0;
    output_params->leave_region_current_frame_boxCount =
        leave_region_current_frame_boxCount0;
    output_params->current_frame_leave_near_fengang_line_right =
        current_frame_leave_near_fengang_line_right0;
    output_params->current_frame_leave_station_number =
        current_frame_leave_station_number0;
    output_params->panjuan_back_state = panjuan_back_state0;

    // 输出前按 y 升序排序（离开区域列表已在绘制前排过序，此处幂等）
    std::sort(boxs_exist_current.begin(), boxs_exist_current.end(), boxMatchCmpYUp);
    std::sort(boxs_leave_current.begin(), boxs_leave_current.end(), boxMatchCmpYUp);

    // 按输出数量截断后写入（列表已存在，越界部分丢弃）
    for (int i = 0;
         i < exist_region_current_frame_boxCount0 &&
         i < (int)boxs_exist_current.size();
         i++) {
        output_params->boxs_exist_region_current[i] = boxs_exist_current[i];
    }
    for (int i = 0;
         i < leave_region_current_frame_boxCount0 &&
         i < (int)boxs_leave_current.size();
         i++) {
        output_params->boxs_leave_region_current[i] = boxs_leave_current[i];
    }

    return true;
}

} // namespace Pipeline
} // namespace JHDeepCore
