#pragma once

#include "reel_diagnostics.h"
#include "reel_types.h"

#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 射线法判断点是否在多边形内（旧版 isPointInPolygon 等价）
bool isPointInPolygon(const ReelPointD &point, const std::vector<ReelPointD> &polygon);

// 排序比较器（旧版 myRect_cmp_* 等价）
bool reelRectCmpXUp(const ReelRect &a, const ReelRect &b);
bool reelRectCmpYUp(const ReelRect &a, const ReelRect &b);
bool reelRectCmpYDown(const ReelRect &a, const ReelRect &b);

// 跨区域重复盘卷过滤（旧版 remove_chongfu_panjuan_between_inside_outside 等价）：
// camera 5 用 camera 3 的输送线内结果过滤自己的输送线外首盘。
// 注意：旧实现里 previous 取 [-1] 下标属未定义行为，这里按其意图取
// 上区域最后一个盘卷 / 起终点数组的第 2 个点。
// decision 为可选出参（默认空）：非空时记录本次判定明细，业务行为不变。
void removeChongfuPanjuanBetweenInsideOutside(const std::vector<ReelRect> &prev_inside,
                                              std::vector<ReelRect> &present,
                                              int &present_count,
                                              const std::vector<ReelPoint> &prev_qishi,
                                              const std::vector<ReelPoint> &present_qishi,
                                              int &guolv_flag,
                                              ReelDedupDecision *decision = nullptr);

} // namespace Pipeline
} // namespace JHDeepCore
