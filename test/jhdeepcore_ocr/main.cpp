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

static cv::Scalar getOCRColor()
{
    return cv::Scalar(0, 255, 0); // 绿色用于OCR
}

static void drawOCRResults(cv::Mat& image, const JHDeepCore::OCRResult& result)
{
    // 获取颜色
    cv::Scalar color = getOCRColor();
    cv::Scalar bgColor = cv::Scalar(0, 0, 0); // 黑色背景

    // 绘制文字边界框（保持原有的框绘制）
    for (const auto& box : result.boxes) {
        if (box.points.empty()) {
            continue;
        }

        try {
            std::vector<cv::Point> points;
            for (const auto& pt : box.points) {
                points.push_back(cv::Point(static_cast<int>(pt.first), static_cast<int>(pt.second)));
            }

            if (points.size() >= 2) {
                cv::polylines(image, points, true, color, 2);
            }
        } catch (...) {
            continue;
        }
    }

    // 将所有识别的文字组合并绘制在左上角
    std::string allText = "OCR Result: ";
    for (size_t i = 0; i < result.boxes.size(); i++) {
        allText += result.boxes[i].text;
        if (i < result.boxes.size() - 1) {
            allText += " "; // 用空格分隔多个文字
        }
    }

    // 在左上角绘制半透明背景
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(allText, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);

    int padding = 10;
    int bgHeight = textSize.height + baseLine + padding * 2;
    int bgWidth = textSize.width + padding * 2;

    cv::Rect bgRect(10, 10, bgWidth, bgHeight);

    // 绘制半透明背景
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, bgRect, bgColor, -1);
    cv::addWeighted(overlay, 0.7, image, 0.3, 0, image);

    // 绘制文字
    cv::putText(image, allText,
               cv::Point(10 + padding, 10 + padding + textSize.height / 2),
               cv::FONT_HERSHEY_SIMPLEX, 0.8,
               cv::Scalar(0, 255, 0), 2); // 绿色文字
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <rec_model> <rec_label> <image_path> [device]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string label_path = argv[2];
    std::string image_path = argv[3];
    std::string device = (argc > 4) ? argv[4] : "cpu";

    int device_id = (device == "cpu") ? -1 : 0;

    try {
        JHDeepCore::OCRRecognizer ocr(model_path, label_path, device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<JHDeepCore::OCRResult> results;
        ocr.process(images, results);

        for (auto& r : results) {
            std::cout << "OCR Result:" << std::endl;
            for (const auto& box : r.boxes) {
                std::cout << "  Text: \"" << box.text << "\" conf=" << box.confidence << std::endl;
            }
        }

        // 创建 result 文件夹（如果不存在）
        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        // 绘制OCR结果
        cv::Mat result_image = image.clone();
        for (const auto& r : results) {
            drawOCRResults(result_image, r);
        }

        // 保存结果
        std::string output_path = "result/ocr.png";
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
