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
struct ReelEventGate {
    bool has_axis = false;
    ReelPoint qishi;
    ReelPoint zhongzhi;
};

// 单次 update 的判定明细（仅用于结果图调试标注，不影响业务输出）
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
//  2. 未匹配的框视为"新卷"，先按最近 dedup_frames 次已上报位置去重
//  3. 入口判定：到 qishi 距离 < entry_radius_px 或 0 <= t < entry_t_ratio
//     （未配置 qishi/zhongzhi 时退化为"新出现即进入"）
//  4. 候选中取到 qishi 距离最小的一个，每相机每次请求最多产生 1 条事件
//  5. 若该相机首次参与请求，或距上次超过 track_timeout_ms，只建立基线
class ReelRegionTracker {
  public:
    explicit ReelRegionTracker(const TedaiJuanquEventConfig &config);

    // 每次请求、每个相机调用一次；产生的事件追加到 events（0 或 1 条）。
    // debug 非空时填充本次判定明细（供结果图标注）。
    void update(int camera_id, const std::string &roi_id, int liuzhi_id,
                const std::vector<ReelRect> &boxes, const ReelEventGate &gate,
                long long now_ms, std::vector<ReelEnterEvent> *events,
                ReelEnterDebug *debug = nullptr);

    void reset();

  private:
    struct Track {
        ReelRect box;
        int missed = 0;
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
