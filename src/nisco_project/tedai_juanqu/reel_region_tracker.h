#pragma once

#include "file_utils.h"
#include "reel_types.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 单个区域的入口判定信息：qishi/zhongzhi 为 ROI 配置里的起点/终点。
// 未配置（点数不足 2）时 has_axis=false，回退为"新出现即进入"。
// 出口感应线（gate_a/gate_b）：过 zhongzhi、垂直于 qishi->zhongzhi 轴的线段，
// 由调用方按 ROI 多边形尺寸推导；has_gate=false 时不做跨线离开判定。
struct ReelEventGate {
    bool has_axis = false;
    ReelPoint qishi;
    ReelPoint zhongzhi;
    bool has_gate = false;
    ReelPoint gate_a;
    ReelPoint gate_b;
};

// 单次 update 的判定明细（仅用于结果图调试标注，不影响业务输出）
struct ReelMotionSeg {
    ReelPoint from;        // 上一帧中心
    ReelPoint to;          // 本帧中心
    bool crossed = false;  // 该线段本次触发了 leave
};

struct ReelEnterDebug {
    enum Rule {
        RULE_NONE = 0,
        RULE_DIST = 1,     // 到 qishi 距离命中
        RULE_T = 2,        // 沿轴投影命中
        RULE_FALLBACK = 3  // 未配置起终点，新出现即进入
    };

    bool baseline = false;       // 本次是否只建基线
    bool has_event = false;
    int new_count = 0;           // 未匹配上的新框数
    int tracked_count = 0;       // 匹配上的老框数
    int dup_count = 0;           // 被去重缓存拦下的新框数
    int left_count = 0;            // 本帧红框总数
    std::vector<ReelRect> left_visible;  // 红框列表：区域内已发 leave 的卷 + 刚出界仍可见的卷
    std::vector<ReelMotionSeg> motion;   // 各匹配轨迹的运动线段（上一帧->本帧）
    Rule rule = RULE_NONE;
    double distance = 0.0;       // 上报框到 qishi 的距离
    double t = 0.0;              // 上报框沿轴的投影比例
    ReelRect box;                // 上报的框（像素坐标）
    std::string event_id;
    ReelEventGate gate;
    std::vector<ReelRect> rejected;    // 未通过入口判定的新框
    std::vector<ReelRect> duplicated;  // 被去重缓存拦下的新框
};

// 盘卷进入区域跟踪器（跨请求状态）。
//
// 判定流程（每个 camera + liuzhi_id）：
//  1. 本帧框与上次请求的框做一对一最近邻匹配（中心距离 < match_radius_px）
//  2. 离开判定：匹配成功的"上一帧->本帧"运动线段向外跨过出口感应线
//     （过 zhongzhi、垂直于轴）时发 1 条 leave 事件，每条轨迹最多 1 条；
//     已报离开且仍被跟踪到的卷通过 debug.left_visible 提供给上层画红框
//  3. 未匹配的框视为"新卷"，先按最近 dedup_frames 次已上报位置去重
//  4. 入口判定：到 qishi 距离 < entry_radius_px 或 0 <= t < entry_t_ratio
//     （未配置 qishi/zhongzhi 时退化为"新出现即进入"）
//  5. 候选中取到 qishi 距离最小的一个，每相机每区域每次请求最多 1 条 enter 事件
//  6. 若该相机首次参与请求，或距上次超过 track_timeout_ms，只建立基线
class ReelRegionTracker {
  public:
    explicit ReelRegionTracker(const TedaiJuanquEventConfig &config);

    // 每次请求、每个相机调用一次；产生的事件追加到 events（enter 每区域最多
    // 1 条，leave 每轨迹最多 1 条）。raw_boxes 为出界标记用的原始检测框，
    // 由 pipeline 预先过滤：仅含中心在本区域多边形外的框；只有已发 leave
    // 的轨迹才会触发单帧红框标记，不续跟。
    // debug 非空时填充本次判定明细（供结果图标注）。
    void update(int camera_id, const std::string &roi_id, int liuzhi_id,
                const std::vector<ReelRect> &boxes, const ReelEventGate &gate,
                long long now_ms, std::vector<ReelEnterEvent> *events,
                ReelEnterDebug *debug = nullptr,
                const std::vector<ReelRect> &raw_boxes = {});

    void reset();

  private:
    struct Track {
        ReelRect box;
        int missed = 0;
        bool fired_leave = false;  // 已发过 leave（每条轨迹最多 1 条）
    };
    struct Reported {
        int x = 0;
        int y = 0;
        int seq = 0;
    };
    struct State {
        std::vector<Track> tracks;
        std::vector<Reported> reported;
        long long last_ms = 0;
        int seq = 0;
    };

    // 返回 true 表示该框满足入口条件；同时给出距离、投影比例与命中规则
    bool entryTest(const ReelRect &box, const ReelEventGate &gate,
                   double *distance, double *t,
                   ReelEnterDebug::Rule *rule) const;

    TedaiJuanquEventConfig config_;
    std::map<std::pair<int, int>, State> states_;  // (camera_id, liuzhi_id)
};

// 生成 evt-<UUID v4> 字符串
std::string makeReelEventId();

} // namespace Pipeline
} // namespace JHDeepCore
