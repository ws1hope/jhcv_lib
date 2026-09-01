#pragma once

#include "reel_types.h"
#include "JHDeepCore.h"

#include <memory>
#include <string>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 基于 ai_platform 统一推理层（JHDeepCore::Detector）的盘卷检测封装。
//
// 说明（2026-09-01 由旧版等价封装切换到 ai_platform，经用户确认接受
// 与旧服务的检测行为差异）：
// - 预处理为标准居中 letterbox（灰 114 填充），坐标经 letterbox 精确
//   反算回原图并截断到图像边界；
// - 置信度/NMS 阈值默认与旧服务一致（conf 0.25 / iou 0.45）；
// - 每个模型需配套同名 YAML（如 panjuan_best_location.yaml）：
//     task_type: detection
//     img_scale: [640, 640]        # 与模型输入一致
//     class_names: [panjuan, abnormal]
//     mean: [0, 0, 0]              # YOLO 仅 /255 归一化
//     std: [255, 255, 255]
//   否则 ai_platform 会使用默认 img_scale(512x512) 与 ImageNet 归一化。
// - 输出 ReelDetObject.rect 的 x/y 转换为中心点坐标：业务层（多边形
//   判定、跨帧匹配、去重、绘制、JSON 输出）的中心点语义保持不变。
class ReelDetector {
  public:
    ReelDetector(const std::string &model_path, bool use_gpu,
                 float conf_threshold = 0.25f, float iou_threshold = 0.45f);

    bool valid() const { return valid_; }

    // 推理一帧；objects 的 rect.x/y 为中心点坐标。
    // 失败（模型未初始化/推理异常）返回 false，objects 置空。
    bool detect(const cv::Mat &frame, std::vector<ReelDetObject> &objects);

  private:
    bool valid_ = false;
    std::unique_ptr<Detector> detector_;
};

// 基于 ai_platform（JHDeepCore::Classifier）的分类封装
// （camera 5 下卷 2 号工位分类，out2_classify.onnx）。
// 预处理为 ai_platform 标准分类流程（resize + ImageNet 均值方差归一化，
// 与旧版 scalefactor≈1/57.3、mean=(123,116,103) 数值上基本一致）。
// 模型需配套同名 YAML（task_type: classification, img_scale: [H, W]）。
class ReelClassifier {
  public:
    struct Result {
        bool valid = false;
        int class_id = -1;
        float confidence = 0.f;
    };

    ReelClassifier(const std::string &model_path, bool use_gpu);

    bool valid() const { return valid_; }

    Result detect(const cv::Mat &frame);

  private:
    bool valid_ = false;
    std::unique_ptr<Classifier> classifier_;
};

} // namespace Pipeline
} // namespace JHDeepCore
