#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <iomanip>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static bool ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

static cv::Scalar getColorForClass(int classId)
{
    static const cv::Scalar colors[] = {
        {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255},
        {255, 0, 255}, {255, 255, 0}, {0, 128, 255}, {255, 128, 0},
        {128, 255, 0}, {0, 255, 128}
    };
    return colors[classId % 10];
}

static void drawDetections(cv::Mat& image, const JHDeepCore::DetectionResult& result)
{
    for (const auto& det : result.detections) {
        cv::Scalar color = getColorForClass(det.class_id);
        cv::rectangle(image, cv::Rect(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height), color, 2);

        std::string label = det.class_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseLine);
        int top = std::max(det.bbox.y, textSize.height + 5);

        cv::rectangle(image, cv::Rect(det.bbox.x, top - textSize.height - 5,
                                      textSize.width + 10, textSize.height + 10), color, -1);
        cv::putText(image, label, cv::Point(det.bbox.x + 5, top - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
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

        const int warmup_runs = 5;
        for (int i = 0; i < warmup_runs; ++i) {
            std::vector<JHDeepCore::DetectionResult> warmup_results;
            detector.process(images, warmup_results);
        }
        std::cout << "Warmup done (" << warmup_runs << " runs)" << std::endl;

        std::vector<JHDeepCore::DetectionResult> results;
        auto t_start = std::chrono::high_resolution_clock::now();
        detector.process(images, results);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "Inference time: " << elapsed_ms << " ms" << std::endl;

        for (auto& r : results) {
            std::cout << "Detection Result:" << std::endl;
            std::cout << "  Num detections: " << r.num_detections << std::endl;
            for (const auto& det : r.detections) {
                std::cout << "  [" << det.class_name << "] conf=" << det.confidence
                          << " bbox=(" << det.bbox.x << "," << det.bbox.y << ","
                          << det.bbox.width << "," << det.bbox.height << ")" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
