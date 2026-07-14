#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "JHDeepCore.h"

namespace JHDeepCore {

/// 九阳语义分割业务模块：封装 JHDeepCore::Segmenter，完成推理并将 mask 叠加到原图
class JiuyangPipeline {
  public:
    /// @param model_path  ONNX 分割模型路径
    /// @param label_path   标签文件路径（可空）
    /// @param device_id    设备 id，>=0 使用 cuda，<0 使用 cpu
    JiuyangPipeline(const std::string &model_path, const std::string &label_path = "",
                    int device_id = 0);

    ~JiuyangPipeline() = default;

    JiuyangPipeline(const JiuyangPipeline &) = delete;
    JiuyangPipeline &operator=(const JiuyangPipeline &) = delete;

    /// 对单张图像执行语义分割，返回叠加了彩色 mask 的结果图
    cv::Mat process(const cv::Mat &image);

    /// 将分割结果（彩色掩码 + 轮廓 + 图例）绘制到 image 上
    static void drawSegmentation(cv::Mat &image, const SegmentationResult &result);

  private:
    std::unique_ptr<Segmenter> segmenter_;
};

} // namespace JHDeepCore
