#pragma once

#include "reel_types.h"

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 旧服务 InferenceYOLOV8DET 的等价封装（第一阶段兼容层）。
// 预处理保持旧版行为：原图贴到左上角黑色正方形画布后缩放到模型输入，
// 输出框 x/y 改写为中心点坐标。为对齐现网检测结果，预处理、阈值
// （conf>0.25 / NMS 0.45）和坐标规则不得修改。
class LegacyReelDetector {
  public:
    LegacyReelDetector(const std::string &model_path, bool use_gpu,
                       int xmin_thresh = 0, int xmax_thresh = 5000,
                       int ymin_thresh = 0, int ymax_thresh = 5000);

    bool valid() const { return valid_; }

    // 推理一帧，objects 的 rect.x/y 为中心点坐标；失败或无目标返回 false/空
    bool detect(const cv::Mat &frame, std::vector<ReelDetObject> &objects);

  private:
    bool valid_ = false;
    cv::Mat input_image_buffer_;  // 复用的画布（旧版内存复用优化）
    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
    std::vector<std::string> input_node_names_str_;
    std::vector<std::string> output_node_names_str_;
    std::vector<const char *> input_node_names_ptr_;
    std::vector<const char *> output_node_names_ptr_;
    int input_w_ = 0;
    int input_h_ = 0;
    int output_h_ = 0;  // 通常 84 = 4 坐标 + 80 类别
    int output_w_ = 0;  // 通常 8400 锚框
    int xmin_thresh_ = 0;
    int xmax_thresh_ = 5000;
    int ymin_thresh_ = 0;
    int ymax_thresh_ = 5000;
};

// 旧服务 InferenceYOLOV8CLS 的等价封装（out2_classify.onnx，
// camera 5 下卷 2 号工位分类）。预处理保持旧版行为：
// scalefactor=1/57.33、mean=(123,116,103)、swapRB。
class LegacyReelClassifier {
  public:
    struct Result {
        bool valid = false;
        int class_id = -1;
        float confidence = 0.f;
    };

    LegacyReelClassifier(const std::string &model_path, bool use_gpu);

    bool valid() const { return valid_; }

    Result detect(const cv::Mat &frame);

  private:
    bool valid_ = false;
    Ort::Env env_{nullptr};
    Ort::Session session_{nullptr};
    std::vector<std::string> input_node_names_str_;
    std::vector<std::string> output_node_names_str_;
    std::vector<const char *> input_node_names_ptr_;
    std::vector<const char *> output_node_names_ptr_;
    int input_w_ = 0;
    int input_h_ = 0;
    int output_w_ = 0;  // 输出形状 [1, num_classes]
};

} // namespace Pipeline
} // namespace JHDeepCore
