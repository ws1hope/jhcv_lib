#include "reel_rules.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace JHDeepCore {
namespace Pipeline {

bool isPointInPolygon(const ReelPointD &point, const std::vector<ReelPointD> &polygon)
{
    int n = static_cast<int>(polygon.size());
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const ReelPointD &p1 = polygon[i];
        const ReelPointD &p2 = polygon[j];
        if ((p1.y > point.y) != (p2.y > point.y)) {
            double intersect_x =
                (point.y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x;
            if (point.x <= intersect_x) {
                inside = !inside;
            }
        }
    }
    return inside;
}

bool reelRectCmpXUp(const ReelRect &a, const ReelRect &b) { return a.x < b.x; }
bool reelRectCmpYUp(const ReelRect &a, const ReelRect &b) { return a.y < b.y; }
bool reelRectCmpYDown(const ReelRect &a, const ReelRect &b) { return a.y > b.y; }

static double reelPointsDistance(double x1, double y1, double x2, double y2)
{
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

void removeChongfuPanjuanBetweenInsideOutside(const std::vector<ReelRect> &prev_inside,
                                              std::vector<ReelRect> &present,
                                              int &present_count,
                                              const std::vector<ReelPoint> &prev_qishi,
                                              const std::vector<ReelPoint> &present_qishi,
                                              int &guolv_flag,
                                              ReelDedupDecision *decision)
{
    if (decision) {
        *decision = ReelDedupDecision();
        decision->prev_inside_count = (int)prev_inside.size();
    }

    if (prev_qishi.size() < 2 || present_qishi.size() < 2) return;
    if (decision) decision->evaluated = true;

    ReelRect previous_last_panjuan_center;  // 默认 (0,0)
    ReelRect present_first_panjuan_center;  // 默认 (0,0)
    if (!prev_inside.empty()) {
        // 旧实现取 [-1] 下标（未定义行为），按意图取最后一个盘卷
        previous_last_panjuan_center = prev_inside.back();
    }
    if (present_count > 0 && !present.empty()) {
        present_first_panjuan_center = present[0];
    }
    if (!prev_inside.empty() && present_count > 0) {
        // 上区域最后盘卷中心 到 上区域终点 的距离
        double previous_distance = reelPointsDistance(
            previous_last_panjuan_center.x, previous_last_panjuan_center.y,
            prev_qishi[1].x, prev_qishi[1].y);
        // 本区域第一盘卷中心 到 本区域起点 的距离
        double present_distance = reelPointsDistance(
            present_first_panjuan_center.x, present_first_panjuan_center.y,
            present_qishi[0].x, present_qishi[0].y);
        if (decision) {
            decision->has_prev_inside = true;
            decision->prev_last_center = previous_last_panjuan_center;
            decision->has_prev_qishi = true;
            decision->prev_qishi_end = prev_qishi[1];
            decision->has_present = true;
            decision->present_first_center = present_first_panjuan_center;
            decision->has_present_qishi = true;
            decision->present_qishi_start = present_qishi[0];
            decision->prev_distance = previous_distance;
            decision->present_distance = present_distance;
        }
        if (previous_distance < 100 && present_distance < 100) {
            if (!present.empty()) {
                present.erase(present.begin());
                present_count--;
                if (decision) {
                    decision->applied = true;
                    decision->removed_count = 1;
                }
            }
            if (present.size() == static_cast<size_t>(present_count)) {
                guolv_flag = 1;
            }
        }
    }
}

} // namespace Pipeline
} // namespace JHDeepCore
