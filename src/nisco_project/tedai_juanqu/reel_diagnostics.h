#pragma once

#include "reel_types.h"

#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 跨区域重复盘卷过滤的判定明细（removeChongfuPanjuan* 的审计输出）。
// 只做记录，不参与业务判定；各字段语义与 reel_rules.cpp 中的计算一一对应。
struct ReelDedupDecision {
    bool evaluated = false;         // 是否进入判定（起点数量不足等会直接跳过）
    bool applied = false;           // 是否实际删除首盘
    bool multi_reel_exempt = false; // 多卷工况命中，跳过过滤
    int removed_count = 0;          // 实际删除数量（当前实现最多 1）

    int prev_camera_id = -1;        // 提供上一区域结果的相机（3 依赖 2、5 依赖 3）
    int prev_inside_count = 0;      // 上一区域输送线内数量

    bool has_prev_inside = false;
    bool has_present = false;
    ReelRect prev_last_center;      // 上一区域最后一个盘卷中心
    ReelRect present_first_center;  // 本区域第一个盘卷中心

    bool has_prev_qishi = false;
    bool has_present_qishi = false;
    ReelPoint prev_qishi_end;       // 上一区域终点（qishi[1]）
    ReelPoint present_qishi_start;  // 本区域起点（qishi[0]）

    double prev_distance = 0.0;     // 上一区域最后盘卷 -> 上一区域终点
    double present_distance = 0.0;  // 本区域首盘 -> 本区域起点
};

// 单个相机一次 process() 的分阶段诊断快照，用于跨帧日志归因：
// 计数变化究竟发生在 检测 / 区域判定 / 跨相机去重 的哪一步。
struct ReelCameraDiagnostics {
    int camera_id = -1;
    bool det_ok = true;                    // 检测是否成功
    std::vector<ReelDetObject> raw;        // 全部原始检测框（含 prob/classid）

    std::vector<ReelRect> inside_assigned;   // 三区域判定后（排序前）
    std::vector<ReelRect> outside_assigned;
    std::vector<ReelRect> collect_assigned;

    int inside_count_before_dedup = 0;
    int inside_count_after_dedup = 0;
    int guolv_flag = 0;          // camera3 去重触发标记
    int guolv_flag_outside = 0;  // camera5 去重触发标记

    ReelDedupDecision dedup_inside;   // camera3：用 camera2 过滤
    ReelDedupDecision dedup_outside;  // camera5：用 camera3 过滤

    int final_inside = 0;    // 最终输出数量（含截断后）
    int final_outside = 0;
    int final_collect = 0;
};

} // namespace Pipeline
} // namespace JHDeepCore
