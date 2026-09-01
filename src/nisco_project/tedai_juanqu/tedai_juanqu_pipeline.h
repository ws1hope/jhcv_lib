#pragma once

#include "reel_types.h"
#include "reel_infer.h"
#include "file_utils.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 特带卷曲多相机盘卷定位编排（旧 HTTP 服务算法核心的迁移）。
//
// 迁移约定（详见 docs/jhcv_lib集成方案.md §0 冻结决策）：
// - 业务规则、处理顺序、坐标语义与旧服务一致；推理层自 2026-09-01
//   起切换为 ai_platform 统一推理（标准 letterbox，检测框坐标与旧版
//   存在合理偏差，已确认接受），模型需配套同名 YAML（见 reel_infer.h）。
// - 一批帧按 camera_id 升序串行处理（camera 3 依赖 camera 2、
//   camera 5 依赖 camera 3 的结果；旧服务隐式依赖请求按升序携带全部相机）。
// - 模型每相机只初始化一次（懒加载）；请求内状态全部局部化，
//   不再有旧服务的全局 zhupiResults/inputparams_history。
// - camera 3 桥下-转弯区域坐标经透视变换映射到物理毫米坐标，
//   其中 x=序号+50 为下游依赖的既有行为，保留。
class TedaiJuanquPipeline {
  public:
    explicit TedaiJuanquPipeline(const TedaiJuanquServerConfig &config);

    // 加载 camera_liuzhi_roi.json（构造时调用，失败返回 false）
    bool loadLiuzhiRois();

    // 一批多相机帧 -> 各相机结果（按 camera_id 升序）
    std::vector<ReelCameraOutput> process(const std::vector<ReelFrameInput> &frames);

    // 预热：按 config 预加载全部相机检测模型与分类模型并各跑一次假推理
    // （触发 ONNX session 创建与 CUDA kernel/显存初始化），避免首个请求
    // 承担加载耗时。任一模型失败只记日志不中断（与懒加载失败行为一致），
    // 全部成功返回 true。
    bool warmup();

    // 本次 process() 内所有相机 ReelDetector::detect / camera5 分类的累计耗时(ms)，
    // 供上层输出每请求一条汇总（process() 每次调用开始时清零）。
    double detectMs() const { return detect_total_ms_; }
    double classifyMs() const { return classify_total_ms_; }

  private:
    ReelDetector *detector(int camera_id);  // 按相机懒加载并缓存
    ReelClassifier *classifier();           // 懒加载并缓存

    // cameras 1-5：检测 + 三区域判定 + 排序 + 跨区域去重 + camera 3 坐标转换
    void detectReelLocation(int camera_id, cv::Mat &src, const ReelRoiParams &params,
                            const ReelCameraOutput *prev_result,
                            const ReelRoiParams *prev_params,
                            ReelCameraOutput &out);

    // camera 0：白色遮挡 + 正常/异常盘卷过滤 + 两区域判定
    void detectReelExport(int camera_id, cv::Mat &src, const ReelRoiParams &params,
                          ReelCameraOutput &out);

    // 结果图保存：<result_dir>\camera_%02d\YYYYMMDD\YYYYMMDDHHMMSSmmm_%02d.jpg（质量 75）
    std::string saveResultImage(int camera_id, const cv::Mat &image);

    TedaiJuanquServerConfig config_;
    std::map<int, std::vector<ReelRoiInfo>> liuzhi_rois_;  // camera_id -> 3 个 ROI
    std::map<int, ReelRoiParams> camera_params_;           // camera_id -> 输入 ROI 参数
    std::map<int, std::unique_ptr<ReelDetector>> detectors_;
    std::unique_ptr<ReelClassifier> classifier_;
    bool classifier_loaded_ = false;
    double detect_total_ms_ = 0;    // 本次 process() 内所有相机检测推理累计 ms
    double classify_total_ms_ = 0;  // 本次 process() 内 camera5 分类推理累计 ms
};

} // namespace Pipeline
} // namespace JHDeepCore
