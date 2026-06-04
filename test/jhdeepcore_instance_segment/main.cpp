#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static bool ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    if (_access(path.c_str(), 0) != 0) {
        return _mkdir(path.c_str()) == 0;
    }
    return true;
#else
    if (access(path.c_str(), F_OK) != 0) {
        return mkdir(path.c_str(), 0755) == 0;
    }
    return true;
#endif
}

static cv::Scalar getColorForClass(int classId)
{
    // 使用更鲜艳、对比度更高的颜色
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
        cv::Scalar(255, 255, 255), // 14: 白色
        cv::Scalar(128, 128, 128)  // 15: 灰色
    };

    if (classId >= 0 && classId < static_cast<int>(colors.size())) {
        return colors[classId];
    }
    // 如果超出预定义颜色范围，使用生成的颜色
    return cv::Scalar(
        (classId * 50) % 256,
        (classId * 100) % 256,
        (classId * 150) % 256
    );
}

static void drawInstanceSegmentation(cv::Mat& image, const JHDeepCore::InstanceSegmentationResult& result)
{
    // 首先绘制边界框和标签
    for (size_t i = 0; i < result.detections.size(); i++) {
        const auto& det = result.detections[i];

        // 获取颜色
        cv::Scalar color = getColorForClass(det.class_id);

        // 绘制边界框（加粗，更明显）
        cv::Rect bbox(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
        cv::rectangle(image, bbox, color, 3);

        // 准备标签文本
        std::string label = det.class_name;
        if (det.confidence > 0) {
            label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        }

        // 计算文本大小
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        // 绘制标签背景（更明显的背景）
        int top = std::max(static_cast<int>(det.bbox.y), textSize.height + 5);
        cv::Rect labelRect(det.bbox.x, top - textSize.height - 5,
                          textSize.width + 10, textSize.height + 10);

        // 半透明背景
        cv::Mat labelOverlay = image.clone();
        cv::rectangle(labelOverlay, labelRect, color, -1);
        cv::addWeighted(labelOverlay, 0.7, image, 0.3, 0, image);

        // 绘制标签文本（白色文字，更大字号）
        cv::putText(image, label,
                   cv::Point(det.bbox.x + 5, top - 3),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   cv::Scalar(255, 255, 255), 2);

        // 如果有对应的掩码且数据类型正确，绘制掩码
        if (i < result.masks.size()) {
            const auto& mask = result.masks[i];
            if (!mask.empty() && mask.type() == CV_32S) {
                // 尝试绘制掩码
                try {
                    cv::Mat maskMat = mask;
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(maskMat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    if (!contours.empty()) {
                        cv::Mat maskOverlay = image.clone();
                        cv::fillPoly(maskOverlay, contours, color);
                        cv::addWeighted(maskOverlay, 0.5, image, 0.5, 0, image);
                    }
                } catch (...) {
                    // 如果掩码绘制失败，跳过
                    continue;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [label_path] [device_id]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = (argc > 3) ? argv[3] : "";
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;

    try {
        JHDeepCore::InstanceSegmenter segmenter(model_path, label_path, device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<JHDeepCore::InstanceSegmentationResult> results;
        segmenter.process(images, results);

        for (auto& r : results) {
            std::cout << "Instance Segmentation Result:" << std::endl;
            std::cout << "  Num detections: " << r.num_detections << std::endl;
            std::cout << "  Num masks: " << r.masks.size() << std::endl;
            for (const auto& det : r.detections) {
                std::cout << "  [" << det.class_name << "] conf=" << det.confidence
                          << " bbox=(" << det.bbox.x << "," << det.bbox.y << ","
                          << det.bbox.width << "," << det.bbox.height << ")" << std::endl;
            }
        }

        // 创建 result 文件夹（如果不存在）
        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        // 绘制实例分割结果
        cv::Mat result_image = image.clone();
        for (const auto& r : results) {
            drawInstanceSegmentation(result_image, r);
        }

        // 保存结果
        std::string output_path = "result/iseg.png";
        if (cv::imwrite(output_path, result_image)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save result to: " << output_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
