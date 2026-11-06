// 盘卷进入区域事件跟踪器单测（不依赖模型/HTTP）
//
// 覆盖：
//  1. 首次请求只建基线，不发事件
//  2. 同一盘卷持续存在，不发事件
//  3. 入口附近新增盘卷 -> 1 条 enter
//  4. 入口附近同时新增两个 -> 只发离 qishi 最近的一条
//  5. 区域中间突然出现 -> 不发
//  6. 漏检后同一盘卷重现 -> 去重缓存生效，不发
//  7. 未配置 qishi（camera4 现状）-> 新出现即进入，取 y 最小
//  8. 超过 track_timeout_ms 未请求 -> 重建基线，不发

#include "reel_region_tracker.h"

#include <cstdio>
#include <string>
#include <vector>

using JHDeepCore::TedaiJuanquEventConfig;
using JHDeepCore::Pipeline::ReelEnterDebug;
using JHDeepCore::Pipeline::ReelEnterEvent;
using JHDeepCore::Pipeline::ReelEventGate;
using JHDeepCore::Pipeline::ReelRect;
using JHDeepCore::Pipeline::ReelRegionTracker;
using JHDeepCore::Pipeline::makeReelEventId;

namespace {

int g_failed = 0;

void expect(bool ok, const char *name)
{
    if (ok) {
        std::printf("[PASS] %s\n", name);
    } else {
        std::printf("[FAIL] %s\n", name);
        ++g_failed;
    }
}

ReelRect makeBox(int x, int y, int w = 100, int h = 100)
{
    ReelRect r;
    r.x = x;
    r.y = y;
    r.width = w;
    r.height = h;
    return r;
}

ReelEventGate makeGate(int qx, int qy, int zx, int zy)
{
    ReelEventGate gate;
    gate.has_axis = true;
    gate.qishi.x = qx;
    gate.qishi.y = qy;
    gate.zhongzhi.x = zx;
    gate.zhongzhi.y = zy;
    return gate;
}

TedaiJuanquEventConfig makeConfig()
{
    TedaiJuanquEventConfig cfg;
    cfg.enable = true;
    cfg.entry_radius_px = 120;
    cfg.entry_t_ratio = 0.25;
    cfg.dedup_frames = 5;
    cfg.track_timeout_ms = 5000;
    cfg.match_radius_px = 150;
    cfg.max_missed = 2;
    return cfg;
}

int runUpdate(ReelRegionTracker &tracker, int camera_id, const char *roi_id,
              int liuzhi_id, const std::vector<ReelRect> &boxes,
              const ReelEventGate &gate, long long now_ms,
              ReelEnterEvent *first_event = nullptr)
{
    std::vector<ReelEnterEvent> events;
    tracker.update(camera_id, roi_id, liuzhi_id, boxes, gate, now_ms, &events);
    if (!events.empty() && first_event) *first_event = events.front();
    return (int)events.size();
}

} // namespace

int main()
{
    const ReelEventGate gate = makeGate(1000, 100, 200, 1000);

    // 1/2) 基线 + 同卷持续
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 1000;
        expect(runUpdate(tracker, 1, "inside", 1, {makeBox(600, 600)}, gate, now) == 0,
               "baseline emits no event");
        now += 100;
        expect(runUpdate(tracker, 1, "inside", 1, {makeBox(610, 600)}, gate, now) == 0,
               "same reel emits no event");
    }

    // 3) 入口附近新增
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 2000;
        runUpdate(tracker, 1, "inside", 1, {makeBox(600, 600)}, gate, now);
        now += 100;
        ReelEnterEvent ev;
        int n = runUpdate(tracker, 1, "inside", 1,
                          {makeBox(600, 600), makeBox(1000, 100)}, gate, now, &ev);
        expect(n == 1 && ev.camera_id == 1 && ev.roi_id == "inside" &&
                   ev.liuzhi_id == 1 && ev.type == "enter" && ev.x == 1000 &&
                   ev.y == 100,
               "new reel near qishi emits one enter");
    }

    // 4) 入口附近同时新增两个：只报离 qishi 最近的一个
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 3000;
        runUpdate(tracker, 1, "inside", 1, {}, gate, now);  // 空基线
        now += 100;
        ReelEnterEvent ev;
        int n = runUpdate(tracker, 1, "inside", 1,
                          {makeBox(1000, 100), makeBox(950, 160)}, gate, now, &ev);
        expect(n == 1 && ev.x == 1000 && ev.y == 100,
               "two new reels near qishi -> only nearest one");
    }

    // 5) 区域中间突然出现：不发
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 4000;
        runUpdate(tracker, 1, "inside", 1, {makeBox(600, 600)}, gate, now);
        now += 100;
        expect(runUpdate(tracker, 1, "inside", 1,
                         {makeBox(600, 600), makeBox(300, 900)}, gate, now) == 0,
               "reel appearing mid-region emits nothing");
    }

    // 6) 漏检后同一盘卷重现：去重缓存生效
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 5000;
        runUpdate(tracker, 1, "inside", 1, {}, gate, now);  // 空基线
        now += 100;
        expect(runUpdate(tracker, 1, "inside", 1, {makeBox(1000, 100)}, gate, now) == 1,
               "first appearance near qishi emits one enter");
        for (int i = 0; i < 4; i++) {
            now += 100;
            runUpdate(tracker, 1, "inside", 1, {}, gate, now);
        }
        now += 100;
        expect(runUpdate(tracker, 1, "inside", 1, {makeBox(1000, 100)}, gate, now) == 0,
               "reappearing reel suppressed by dedup cache");
    }

    // 7) camera4 outside：未配置 qishi -> 新出现即进入，取 y 最小
    {
        ReelRegionTracker tracker(makeConfig());
        ReelEventGate no_axis;
        long long now = 6000;
        expect(runUpdate(tracker, 4, "outside", 2, {makeBox(300, 700)}, no_axis,
                         now) == 0,
               "cam4 baseline emits no event");
        now += 100;
        ReelEnterEvent ev;
        int n = runUpdate(tracker, 4, "outside", 2,
                          {makeBox(300, 700), makeBox(300, 500), makeBox(320, 620)},
                          no_axis, now, &ev);
        expect(n == 1 && ev.camera_id == 4 && ev.roi_id == "outside" &&
                   ev.liuzhi_id == 2 && ev.y == 500,
               "cam4 fallback picks smallest y");
    }

    // 8) 超过 track_timeout_ms：重建基线，不发事件
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 7000;
        runUpdate(tracker, 1, "inside", 1, {makeBox(600, 600)}, gate, now);
        now += 6000;  // > 5000ms
        expect(runUpdate(tracker, 1, "inside", 1,
                         {makeBox(600, 600), makeBox(1000, 100)}, gate, now) == 0,
               "timeout rebuilds baseline without events");
    }

    // 9) 调试信息：命中规则 / 距离 / 投影比例
    {
        ReelRegionTracker tracker(makeConfig());
        long long now = 8000;
        std::vector<ReelEnterEvent> events;
        ReelEnterDebug dbg;
        tracker.update(1, "inside", 1, {}, gate, now, &events, &dbg);
        expect(dbg.baseline, "debug marks baseline");

        now += 100;
        tracker.update(1, "inside", 1, {makeBox(1000, 100)}, gate, now, &events, &dbg);
        expect(dbg.has_event && dbg.rule == ReelEnterDebug::RULE_DIST &&
                   dbg.distance < 1.0,
               "debug reports dist rule");

        now += 100;
        std::vector<ReelEnterEvent> ev2;
        tracker.update(1, "inside", 1,
                       {makeBox(1000, 100), makeBox(880, 235)}, gate, now, &ev2,
                       &dbg);
        expect(dbg.has_event && dbg.rule == ReelEnterDebug::RULE_T &&
                   dbg.t > 0.0 && dbg.t < 0.25,
               "debug reports t rule");
    }

    // event_id 形状与唯一性
    {
        std::string id = makeReelEventId();
        expect(id.rfind("evt-", 0) == 0 && id.size() == 40,
               "event id has evt-uuid format");
        std::string id2 = makeReelEventId();
        expect(id != id2, "event ids are unique");
    }

    std::printf("\n%s (%d failure(s))\n", g_failed == 0 ? "ALL PASS" : "FAILED",
                g_failed);
    return g_failed == 0 ? 0 : 1;
}
