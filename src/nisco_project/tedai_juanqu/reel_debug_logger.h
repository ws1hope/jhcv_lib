#pragma once

#include "reel_diagnostics.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace JHDeepCore {
namespace Pipeline {

// tedai_juanqu 分阶段调试日志（JSONL，每相机每请求一行）。
//
// 目的：不改变任何业务行为的前提下，把「检测 -> 区域判定 -> 跨相机去重 ->
// 最终输出」各阶段的框与计数落盘，供离线脚本定位「目标丢失/计数抖动」发生在
// 哪一步（漏检 / ROI 边界 / 去重 / 其它）。
//
// 文件：<log_dir>/tedai_debug_YYYYMMDD.jsonl（按天切换，追加写）。
// 关闭时（enabled=false）不创建文件、不产生任何副作用。
class ReelDebugLogger {
  public:
    ReelDebugLogger(bool enabled, const std::string &log_dir);

    bool enabled() const { return enabled_; }

    // 写一条相机级诊断记录。camera_id/read_picture_flag/diag 均来自本次请求。
    void logCamera(const std::string &station_id, uint64_t req_index,
                   int read_picture_flag, const ReelCameraDiagnostics &diag);

  private:
    void ensureOpenForToday();

    bool enabled_ = false;
    std::string log_dir_;
    std::string current_date_;
    std::ofstream stream_;
};

} // namespace Pipeline
} // namespace JHDeepCore
