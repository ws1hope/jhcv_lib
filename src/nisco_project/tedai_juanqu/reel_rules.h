#pragma once

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

// 相邻区域重复盘卷过滤（旧版 remove_chongfu_panjuan 等价）：
// camera 3 用 camera 2 的输送线内结果过滤自己的输送线内首盘。
// prev_qishi/present_qishi 为两个区域的起终点（各 2 个点）。
// 多卷工况（上个区域 >1 盘且本区域 >10 盘、间距特征匹配）时跳过过滤。
void removeChongfuPanjuan(const std::vector<ReelRect> &prev_inside,
                          std::vector<ReelRect> &present, int &present_count,
                          const std::vector<ReelPoint> &prev_qishi,
                          const std::vector<ReelPoint> &present_qishi,
                          int &guolv_flag);

// 跨区域重复盘卷过滤（旧版 remove_chongfu_panjuan_between_inside_outside 等价）：
// camera 5 用 camera 3 的输送线内结果过滤自己的输送线外首盘。
// 注意：旧实现里 previous 取 [-1] 下标属未定义行为，这里按其意图取
// 上区域最后一个盘卷 / 起终点数组的第 2 个点。
void removeChongfuPanjuanBetweenInsideOutside(const std::vector<ReelRect> &prev_inside,
                                              std::vector<ReelRect> &present,
                                              int &present_count,
                                              const std::vector<ReelPoint> &prev_qishi,
                                              const std::vector<ReelPoint> &present_qishi,
                                              int &guolv_flag);

// 透视变换：像素中心点 -> 物理毫米坐标（旧版 transform_rects_to_physical 等价）。
// 输出 x 被置为 序号+50、y 取 max(0, 四舍五入)，location=1——
// x=i+50 为旧服务既有行为，下游依赖该坐标，勿改。
std::vector<ReelRect> transformRectsToPhysical(
    const std::vector<ReelRect> &input_rects,
    const std::vector<cv::Point2f> &calibration_src_points,
    float real_w_mm, float real_h_mm);

} // namespace Pipeline
} // namespace JHDeepCore
