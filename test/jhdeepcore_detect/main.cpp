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
    // 为不同类别生成不同的颜色
    return cv::Scalar(
        (classId * 37) % 256,
        (classId * 67) % 256,
        (classId * 97) % 256
    );
}

static void drawDetections(cv::Mat& image, const JHDeepCore::DetectionResult& result)
{
    for (const auto& det : result.detections) {
        // 获取颜色
        cv::Scalar color = getColorForClass(det.class_id);

        // 绘制边界框
        cv::Rect bbox(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
        cv::rectangle(image, bbox, color, 2);

        // 准备标签文本
        std::string label = det.class_name;
        if (det.confidence > 0) {
            label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        }

        // 计算文本大小
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        // 绘制标签背景
        int top = std::max(static_cast<int>(det.bbox.y), textSize.height + 5);
        cv::Rect labelRect(det.bbox.x, top - textSize.height - 5,
                          textSize.width + 10, textSize.height + 10);
        cv::rectangle(image, labelRect, color, -1);

        // 绘制标签文本
        cv::putText(image, label,
                   cv::Point(det.bbox.x + 5, top - 3),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6,
                   cv::Scalar(255, 255, 255), 1);
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
        JHDeepCore::Detector detector(model_path, label_path, device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<JHDeepCore::DetectionResult> results;
        detector.process(images, results);

        // 打印检测结果
        for (auto& r : results) {
            std::cout << "Detection Result:" << std::endl;
            std::cout << "  Num detections: " << r.num_detections << std::endl;
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

        // 绘制检测结果
        cv::Mat result_image = image.clone();
        for (const auto& r : results) {
            drawDetections(result_image, r);
        }

        // 保存结果
        std::string output_path = "result/det.png";
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
