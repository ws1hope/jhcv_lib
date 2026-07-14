#include "jiuyang_pipeline.h"

#include <stdexcept>

namespace JHDeepCore {

JiuyangPipeline::JiuyangPipeline(const std::string &model_path, const std::string &label_path,
                                 int device_id)
    : segmenter_(std::make_unique<Segmenter>(model_path, label_path, device_id)) {}

cv::Mat JiuyangPipeline::process(const cv::Mat &image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    std::vector<cv::Mat> images = {image};
    std::vector<SegmentationResult> results;
    segmenter_->process(images, results);

    cv::Mat result_image = image.clone();
    for (const auto &r : results) {
        drawSegmentation(result_image, r);
    }
    return result_image;
}

static cv::Scalar getColorForClass(int classId) {
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 0),        // 0: 背景 - 黑色
        cv::Scalar(255, 0, 0),      // 1: 红色
        cv::Scalar(0, 255, 0),      // 2: 绿色
        cv::Scalar(0, 0, 255),      // 3: 蓝色
        cv::Scalar(255, 255, 0),    // 4: 青色
        cv::Scalar(255, 0, 255),    // 5: 品红色
        cv::Scalar(0, 255, 255),    // 6: 黄色
        cv::Scalar(128, 0, 128),    // 7: 紫色
        cv::Scalar(255, 128, 0),    // 8: 橙色
        cv::Scalar(0, 128, 255),    // 9: 天蓝色
        cv::Scalar(128, 255, 0),    // 10: 黄绿色
        cv::Scalar(255, 0, 128),    // 11: 玫瑰红
        cv::Scalar(128, 0, 255),    // 12: 紫罗兰
        cv::Scalar(0, 255, 128),    // 13: 薄荷绿
        cv::Scalar(255, 255, 255),  // 14: 白色
        cv::Scalar(128, 128, 128)   // 15: 灰色
    };

    if (classId >= 0 && classId < static_cast<int>(colors.size())) {
        return colors[classId];
    }
    return cv::Scalar((classId * 50) % 256, (classId * 100) % 256, (classId * 150) % 256);
}

void JiuyangPipeline::drawSegmentation(cv::Mat &image, const SegmentationResult &result) {
    if (result.segmentation_mask.empty()) {
        return;
    }

    // 创建彩色分割掩码
    cv::Mat coloredMask = cv::Mat::zeros(result.segmentation_mask.size(), CV_8UC3);

    for (int i = 0; i < result.num_classes; i++) {
        cv::Scalar color = getColorForClass(i);
        cv::Mat classMask = (result.segmentation_mask == i);
        coloredMask.setTo(color, classMask);

        // 为每个类别添加边界轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat tempMask;
        classMask.copyTo(tempMask);
        cv::findContours(tempMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 在原图上绘制轮廓，增强可视化效果
        cv::drawContours(image, contours, -1, color, 2);
    }

    // 使用更强的对比度进行叠加
    cv::Mat overlay;
    cv::addWeighted(image, 0.4, coloredMask, 0.6, 0, overlay);
    overlay.copyTo(image);

    // 添加类别图例
    int legendX = 20;
    int legendY = 20;
    int boxSize = 20;
    int spacing = 30;

    // 绘制图例背景
    int legendHeight = (result.num_classes * spacing) + 20;
    cv::Rect legendBg(10, 10, 150, legendHeight);
    cv::Mat legendOverlay = image.clone();
    cv::rectangle(legendOverlay, legendBg, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(legendOverlay, 0.7, image, 0.3, 0, image);

    // 绘制类别标签
    for (int i = 0; i < result.num_classes; i++) {
        cv::Scalar color = getColorForClass(i);

        // 绘制颜色方块
        cv::Rect colorBox(legendX, legendY + (i * spacing), boxSize, boxSize);
        cv::rectangle(image, colorBox, color, -1);
        cv::rectangle(image, colorBox, cv::Scalar(255, 255, 255), 1);

        // 绘制类别名称
        std::string label = "Class " + std::to_string(i);
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(image, label,
                    cv::Point(legendX + boxSize + 10, legendY + (i * spacing) + boxSize / 2 + baseLine / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // 生成与原图同等大小的二值图像（前景白色，背景黑色）
    cv::Mat binaryMask;
    cv::threshold(result.segmentation_mask, binaryMask, 0, 255, cv::THRESH_BINARY);
    cv::Mat binaryColor;
    cv::cvtColor(binaryMask, binaryColor, cv::COLOR_GRAY2BGR);
    if (binaryColor.size() != image.size()) {
        cv::resize(binaryColor, binaryColor, image.size());
    }

    // 添加标题
    cv::putText(image, "Mask Overlay", cv::Point(20, image.rows - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(binaryColor, "Binary Mask", cv::Point(20, binaryColor.rows - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 水平拼接：左 mask 叠加图，右二值图
    cv::Mat combined;
    cv::hconcat(image, binaryColor, combined);
    combined.copyTo(image);
}

} // namespace JHDeepCore
