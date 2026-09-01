#pragma once

#include "reel_types.h"
#include "reel_legacy_infer.h"
#include "reel_track_dll.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 盘卷出口单帧跟踪（旧 DLL `MV_SDK_ReelExportTargetTrack_Detect_2_Impl`
// 的等价迁移，C# P/Invoke 直调路径的算法核心）。
//
// 等价迁移约定：
// - 阈值、分支、输出字段与旧版一致；输入输出直接使用 ABI 结构体。
// - 旧版 Detect_2_Impl 函数内 static 的报警跟踪状态
//   （g_prev_station_one/two_real_state、g_prev_reels）移到实例成员；
//   DLL 线上单路相机，单实例使用，重新 Init/Destroy 时状态复位。
// - 3 通道输入时 src_image 包裹调用方缓冲区，绘制结果会原地修改
//   调用方位图数据（旧版行为，见 reel_track_dll.h 说明）。
// - 旧版死代码不迁移：change_state 跳变消抖计算（算完从未使用）、
//   right_filter_points 新进入盘卷统计（仅用于 cout）。
class ReelTrackPipeline {
  public:
    ReelTrackPipeline() = default;
    ~ReelTrackPipeline() = default;

    bool init(const std::string& model_path, bool use_gpu,
              int xmin_thresh, int xmax_thresh, int ymin_thresh, int ymax_thresh);
    bool valid() const { return detector_ && detector_->valid(); }

    // 单帧跟踪：src_image 为输入图（会被绘制结果）；
    // 返回 false 表示内部错误（模型未初始化等），output_params 不保证有效。
    bool detect(cv::Mat &src_image,
                const InputParamsReelExportTargetTrack &input_params,
                OutputResultReelExportTargetTrack *output_params);

  private:
    std::unique_ptr<LegacyReelDetector> detector_;
    // 异常卷边缘触发报警的跨帧状态（旧版函数内 static）
    int prev_station_one_real_state_ = 0;
    int prev_station_two_real_state_ = 0;
    std::vector<std::pair<int, int>> prev_reels_;  // (中心x, 出生工位)
};

} // namespace Pipeline
} // namespace JHDeepCore
