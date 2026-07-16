#include <iostream>
#include <filesystem>
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

static void drawSegmentation(cv::Mat& image, const JHDeepCore::SegmentationResult& result)
{
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

    // 将结果复制回原始图像
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
                   cv::Point(legendX + boxSize + 10, legendY + (i * spacing) + boxSize/2 + baseLine/2),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                   cv::Scalar(255, 255, 255), 1);
    }
}

static std::string makeOutputPathFromInput(const std::string& image_path)
{
    std::string filename = std::filesystem::path(image_path).filename().string();
    return "result/" + filename;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [label_path] [device_id]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = "";
    std::string device = "cpu";

    if (argc > 3) {
        std::string arg3 = argv[3];
        if (arg3 == "cpu" || arg3 == "gpu" || arg3 == "cuda") {
            device = arg3;
        } else {
            label_path = arg3;
            if (argc > 4) {
                device = argv[4];
            }
        }
    }

    int device_id = (device == "cpu") ? -1 : 0;

    try {
        JHDeepCore::Segmenter segmenter(model_path, label_path, device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<JHDeepCore::SegmentationResult> results;
        segmenter.process(images, results);

        for (auto& r : results) {
            std::cout << "Segmentation Result:" << std::endl;
            std::cout << "  Num classes: " << r.num_classes << std::endl;
            std::cout << "  Mask size: " << r.segmentation_mask.cols << "x" << r.segmentation_mask.rows << std::endl;
        }

        // 创建 result 文件夹（如果不存在）
        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        // 绘制分割结果
        cv::Mat result_image = image.clone();
        for (const auto& r : results) {
            drawSegmentation(result_image, r);
        }

        // 保存结果（按输入图片文件名保存到 result/ 目录）
        std::string output_path = makeOutputPathFromInput(image_path);
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
