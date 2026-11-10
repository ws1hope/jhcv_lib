#include "reel_region_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>

namespace JHDeepCore {
namespace Pipeline {

namespace {

double rectCenterDistanceSq(const ReelRect &a, const ReelRect &b)
{
    double dx = (double)a.x - (double)b.x;
    double dy = (double)a.y - (double)b.y;
    return dx * dx + dy * dy;
}

int signOf(double v)
{
    return (v > 0.0) - (v < 0.0);
}

// 离开判定（跨线检测）：匹配成功的运动线段 prev->cur 是否从 qishi 一侧
// 向外跨过出口感应线 gate_a->gate_b。交点须落在感应线段范围内；
// prev 恰好压线时允许继续向外跨出（s1 != 外侧且 s2 在外侧）。
bool crossingExitGate(const ReelRect &prev, const ReelRect &cur,
                      const ReelEventGate &gate)
{
    const double ax = (double)gate.gate_a.x;
    const double ay = (double)gate.gate_a.y;
    const double gx = (double)gate.gate_b.x - ax;
    const double gy = (double)gate.gate_b.y - ay;
    if (std::abs(gx) < 1e-9 && std::abs(gy) < 1e-9) return false;

    // d = cross(g, P-A)：P 相对感应线的侧别
    const double d1 =
        gx * ((double)prev.y - ay) - gy * ((double)prev.x - ax);
    const double d2 =
        gx * ((double)cur.y - ay) - gy * ((double)cur.x - ax);
    const double dref = gx * ((double)gate.qishi.y - ay) -
                        gy * ((double)gate.qishi.x - ax);
    const int s1 = signOf(d1);
    const int s2 = signOf(d2);
    const int sref = signOf(dref);
    if (sref == 0) return false;
    if (s1 == sref && s2 == 0) return false;  // 恰好压线，尚未跨出
    if (s1 == -sref || s2 != -sref) return false;  // 方向不对或未跨线

    // 交点沿感应线的参数 s 须在 [0,1] 内（denom!=0 由 d1/d2 异号保证）
    const double mx = (double)cur.x - (double)prev.x;
    const double my = (double)cur.y - (double)prev.y;
    const double denom = gx * my - gy * mx;
    if (std::abs(denom) < 1e-9) return false;
    const double s =
        (((double)prev.x - ax) * my - ((double)prev.y - ay) * mx) / denom;
    return s >= 0.0 && s <= 1.0;
}

} // namespace

ReelRegionTracker::ReelRegionTracker(const TedaiJuanquEventConfig &config)
    : config_(config)
{
}

void ReelRegionTracker::reset()
{
    states_.clear();
}

bool ReelRegionTracker::entryTest(const ReelRect &box, const ReelEventGate &gate,
                                  double *distance, double *t,
                                  ReelEnterDebug::Rule *rule) const
{
    if (distance) *distance = 0.0;
    if (t) *t = 0.0;
    if (rule) *rule = ReelEnterDebug::RULE_NONE;

    if (!gate.has_axis) {
        // 未配置起终点：退化为"新出现即进入"
        if (rule) *rule = ReelEnterDebug::RULE_FALLBACK;
        return true;
    }

    double dx = (double)box.x - (double)gate.qishi.x;
    double dy = (double)box.y - (double)gate.qishi.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    if (distance) *distance = dist;

    // 沿 qishi->zhongzhi 轴投影：t=0 在起点，t=1 在终点
    double ax = (double)gate.zhongzhi.x - (double)gate.qishi.x;
    double ay = (double)gate.zhongzhi.y - (double)gate.qishi.y;
    double len2 = ax * ax + ay * ay;
    double proj = 0.0;
    if (len2 > 0.0) {
        proj = (dx * ax + dy * ay) / len2;
        if (t) *t = proj;
    }

    if (dist < (double)config_.entry_radius_px) {
        if (rule) *rule = ReelEnterDebug::RULE_DIST;
        return true;
    }
    if (len2 > 0.0 && proj >= 0.0 && proj < config_.entry_t_ratio) {
        if (rule) *rule = ReelEnterDebug::RULE_T;
        return true;
    }
    return false;
}

void ReelRegionTracker::update(int camera_id, const std::string &roi_id,
                               int liuzhi_id, const std::vector<ReelRect> &boxes,
                               const ReelEventGate &gate, long long now_ms,
                               std::vector<ReelEnterEvent> *events,
                               ReelEnterDebug *debug)
{
    if (debug) {
        *debug = ReelEnterDebug();
        debug->gate = gate;
    }
    if (!config_.enable) return;

    State &st = states_[std::make_pair(camera_id, liuzhi_id)];
    st.seq++;

    // 首次参与 / 长时间未参与：只建立基线，不发事件
    bool baseline = (st.last_ms == 0) ||
                    (now_ms - st.last_ms > (long long)config_.track_timeout_ms);
    st.last_ms = now_ms;
    if (baseline) {
        st.tracks.clear();
        st.reported.clear();
        for (const auto &box : boxes) {
            st.tracks.push_back(Track{box, 0});
        }
        if (debug) {
            debug->baseline = true;
            debug->tracked_count = (int)boxes.size();
        }
        return;
    }

    // 1) 与上次请求的框做一对一最近邻匹配；匹配成功的做跨线离开判定
    const double match_r2 =
        (double)config_.match_radius_px * (double)config_.match_radius_px;
    std::vector<bool> box_used(boxes.size(), false);
    std::vector<Track> next_tracks;
    int matched_count = 0;
    for (auto &track : st.tracks) {
        int best = -1;
        double best_d2 = match_r2;
        for (size_t i = 0; i < boxes.size(); i++) {
            if (box_used[i]) continue;
            double d2 = rectCenterDistanceSq(track.box, boxes[i]);
            if (d2 <= best_d2) {
                best_d2 = d2;
                best = (int)i;
            }
        }
        if (best >= 0) {
            const ReelRect prev_box = track.box;
            const ReelRect &cur_box = boxes[best];

            // 离开判定：运动线段向外跨过出口感应线，每条轨迹最多 1 条 leave
            if (gate.has_gate && !track.fired_leave &&
                crossingExitGate(prev_box, cur_box, gate)) {
                track.fired_leave = true;
                ReelEnterEvent ev;
                ev.event_id = makeReelEventId();
                ev.type = "leave";
                ev.camera_id = camera_id;
                ev.roi_id = roi_id;
                ev.liuzhi_id = liuzhi_id;
                ev.x = cur_box.x;
                ev.y = cur_box.y;
                ev.width = cur_box.width;
                ev.height = cur_box.height;
                if (events) events->push_back(ev);
            }

            // 已报离开且仍被跟踪到的卷：交给上层在结果图画红框
            if (track.fired_leave) {
                if (debug) {
                    debug->left_count++;
                    debug->left_visible.push_back(cur_box);
                }
            }

            track.box = cur_box;
            track.missed = 0;
            box_used[best] = true;
            next_tracks.push_back(track);
            matched_count++;
        } else {
            track.missed++;
            if (track.missed <= config_.max_missed) next_tracks.push_back(track);
        }
    }

    if (debug) {
        debug->tracked_count = matched_count;
        debug->new_count = (int)boxes.size() - matched_count;
    }

    // 2) 淘汰过期的"已上报位置"缓存
    st.reported.erase(
        std::remove_if(st.reported.begin(), st.reported.end(),
                       [&](const Reported &r) {
                           return st.seq - r.seq > config_.dedup_frames;
                       }),
        st.reported.end());

    // 3) 新卷：去重缓存过滤 + 入口判定，选离 qishi 最近的一个
    int best_new = -1;
    double best_new_distance = 0.0;
    double best_new_t = 0.0;
    ReelEnterDebug::Rule best_new_rule = ReelEnterDebug::RULE_NONE;
    const double dedup_r2 =
        (double)config_.entry_radius_px * (double)config_.entry_radius_px;
    for (size_t i = 0; i < boxes.size(); i++) {
        if (box_used[i]) continue;
        const ReelRect &box = boxes[i];

        bool duplicated = false;
        for (const auto &r : st.reported) {
            double dx = (double)box.x - (double)r.x;
            double dy = (double)box.y - (double)r.y;
            if (dx * dx + dy * dy <= dedup_r2) {
                duplicated = true;
                break;
            }
        }
        if (duplicated) {
            if (debug) {
                debug->dup_count++;
                debug->duplicated.push_back(box);
            }
            continue;
        }

        double distance = 0.0;
        double t = 0.0;
        ReelEnterDebug::Rule rule = ReelEnterDebug::RULE_NONE;
        if (!entryTest(box, gate, &distance, &t, &rule)) {
            if (debug) debug->rejected.push_back(box);
            continue;
        }

        if (best_new < 0) {
            best_new = (int)i;
            best_new_distance = distance;
            best_new_t = t;
            best_new_rule = rule;
        } else if (gate.has_axis) {
            if (distance < best_new_distance) {
                best_new = (int)i;
                best_new_distance = distance;
                best_new_t = t;
                best_new_rule = rule;
            }
        } else if (box.y < boxes[best_new].y) {
            // 无起终点配置时回退：取 y 最小的新卷
            best_new = (int)i;
            best_new_distance = distance;
            best_new_t = t;
            best_new_rule = rule;
        }
    }

    // 4) enter 事件：每相机每区域每次请求最多 1 条（leave 事件在匹配阶段产生，不占此名额）
    if (best_new >= 0) {
        const ReelRect &box = boxes[best_new];
        ReelEnterEvent ev;
        ev.event_id = makeReelEventId();
        ev.type = "enter";
        ev.camera_id = camera_id;
        ev.roi_id = roi_id;
        ev.liuzhi_id = liuzhi_id;
        ev.x = box.x;
        ev.y = box.y;
        ev.width = box.width;
        ev.height = box.height;
        if (events) events->push_back(ev);
        st.reported.push_back(Reported{box.x, box.y, st.seq});

        if (debug) {
            debug->has_event = true;
            debug->box = box;
            debug->distance = best_new_distance;
            debug->t = best_new_t;
            debug->rule = best_new_rule;
            debug->event_id = ev.event_id;
        }
    }

    // 5) 新出现的框全部纳入轨迹（避免下一帧重复判定为"进入"）
    for (size_t i = 0; i < boxes.size(); i++) {
        if (!box_used[i]) {
            next_tracks.push_back(Track{boxes[i], 0});
        }
    }
    st.tracks.swap(next_tracks);
}

std::string makeReelEventId()
{
    static std::mutex rng_mutex;
    static std::mt19937_64 rng([]() {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    }());

    unsigned char bytes[16];
    {
        std::lock_guard<std::mutex> lock(rng_mutex);
        uint64_t a = rng();
        uint64_t b = rng();
        std::memcpy(bytes, &a, sizeof(a));
        std::memcpy(bytes + sizeof(a), &b, sizeof(b));
    }
    bytes[6] = (unsigned char)((bytes[6] & 0x0F) | 0x40);  // UUID v4
    bytes[8] = (unsigned char)((bytes[8] & 0x3F) | 0x80);  // variant

    char buf[64];
    std::snprintf(
        buf, sizeof(buf),
        "evt-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
        bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12],
        bytes[13], bytes[14], bytes[15]);
    return std::string(buf);
}

} // namespace Pipeline
} // namespace JHDeepCore
