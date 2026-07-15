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

    last_bend_results_.clear();
    cv::Mat bend_overlay = image.clone();
    cv::Mat result_image = image.clone();
    for (const auto &r : results) {
        if (!r.segmentation_mask.empty()) {
            cv::Mat binary_mask;
            cv::compare(r.segmentation_mask, 0, binary_mask, cv::CMP_GT);
            if (binary_mask.size() != image.size()) {
                cv::resize(binary_mask, binary_mask, image.size(), 0, 0, cv::INTER_NEAREST);
            }
            last_bend_results_ = bend_detector_.detect(binary_mask);
            BilletBendDetector::draw(bend_overlay, last_bend_results_);
        }
        drawSegmentation(result_image, r);
    }

    // drawSegmentation currently returns a side-by-side image. Put bend analysis on
    // its left (original-image) panel while retaining the binary-mask panel.
    if (result_image.cols >= image.cols && result_image.rows >= image.rows) {
        bend_overlay.copyTo(result_image(cv::Rect(0, 0, image.cols, image.rows)));
    } else {
        result_image = bend_overlay;
    }
    return result_image;
}

const std::vector<BilletBendResult> &JiuyangPipeline::lastBendResults() const noexcept {
    return last_bend_results_;
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

static cv::Mat zhangSuenThinning(const cv::Mat &binaryIn) {
    cv::Mat src;
    cv::threshold(binaryIn, src, 0, 255, cv::THRESH_BINARY);
    src.convertTo(src, CV_8UC1);

    cv::Mat cur = src.clone();
    const int maxIter = 1000;

    for (int iter = 0; iter < maxIter; ++iter) {
        bool changed = false;

        for (int sub = 0; sub < 2; ++sub) {
            cv::Mat marker = cv::Mat::zeros(cur.size(), CV_8UC1);

            for (int y = 1; y < cur.rows - 1; ++y) {
                const uchar *row = cur.ptr<uchar>(y);
                uchar *mk = marker.ptr<uchar>(y);
                for (int x = 1; x < cur.cols - 1; ++x) {
                    if (row[x] == 0) continue;

                    const int p2 = cur.at<uchar>(y - 1, x) > 0 ? 1 : 0;
                    const int p3 = cur.at<uchar>(y - 1, x + 1) > 0 ? 1 : 0;
                    const int p4 = cur.at<uchar>(y, x + 1) > 0 ? 1 : 0;
                    const int p5 = cur.at<uchar>(y + 1, x + 1) > 0 ? 1 : 0;
                    const int p6 = cur.at<uchar>(y + 1, x) > 0 ? 1 : 0;
                    const int p7 = cur.at<uchar>(y + 1, x - 1) > 0 ? 1 : 0;
                    const int p8 = cur.at<uchar>(y, x - 1) > 0 ? 1 : 0;
                    const int p9 = cur.at<uchar>(y - 1, x - 1) > 0 ? 1 : 0;

                    const int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    if (B < 2 || B > 6) continue;

                    const int seq[9] = {p2, p3, p4, p5, p6, p7, p8, p9, p2};
                    int A = 0;
                    for (int k = 0; k < 8; ++k) {
                        if (seq[k] == 0 && seq[k + 1] == 1) ++A;
                    }
                    if (A != 1) continue;

                    if (sub == 0) {
                        if (p2 * p4 * p6 != 0) continue;
                        if (p4 * p6 * p8 != 0) continue;
                    } else {
                        if (p2 * p4 * p8 != 0) continue;
                        if (p2 * p6 * p8 != 0) continue;
                    }

                    mk[x] = 1;
                }
            }

            for (int y = 1; y < cur.rows - 1; ++y) {
                uchar *c = cur.ptr<uchar>(y);
                const uchar *mk = marker.ptr<uchar>(y);
                for (int x = 1; x < cur.cols - 1; ++x) {
                    if (mk[x]) {
                        c[x] = 0;
                        changed = true;
                    }
                }
            }
        }

        if (!changed) break;
    }

    return cur;
}

/// 基于距离变换提取二值掩码的中心线（单像素宽）
static cv::Mat extractCenterline(const cv::Mat &inputMask) {
    cv::Mat binary;
    if (inputMask.channels() == 3) {
        cv::cvtColor(inputMask, binary, cv::COLOR_BGR2GRAY);
    } else {
        binary = inputMask.clone();
    }
    cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);

    // 1. 距离变换
    cv::Mat dist;
    cv::distanceTransform(binary, dist, cv::DIST_L2, cv::DIST_MASK_5);

    // 2. 3x3 邻域最大值
    cv::Mat dilated;
    cv::dilate(dist, dilated, cv::Mat());

    // 3. 提取局部最大值（脊线）
    cv::Mat ridgeMask;
    cv::compare(dist, dilated, ridgeMask, cv::CMP_GE);

    // 4. 限制在原始前景内部
    cv::bitwise_and(ridgeMask, binary, ridgeMask);

    // 5. 去掉过于靠近边缘的点
    cv::Mat validDistanceMask;
    cv::compare(dist, 1.5f, validDistanceMask, cv::CMP_GT);
    cv::bitwise_and(ridgeMask, validDistanceMask, ridgeMask);

    // 6. 细化成单像素线
    cv::Mat centerline = zhangSuenThinning(ridgeMask);

    return centerline;
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

    // 提取中心线并叠加到二值图上（红色）
    cv::Mat centerline = extractCenterline(binaryMask);
    if (centerline.size() == binaryColor.size()) {
        binaryColor.setTo(cv::Scalar(0, 0, 255), centerline);
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
