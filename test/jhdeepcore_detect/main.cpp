#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
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
    std::string model_path = (argc > 1) ? argv[1] : "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/tt/best.onnx";
    std::string image_dir = (argc > 2) ? argv[2] : "/Users/zhanghaining/2026code/jhcv_lib/images/multibatch";
    std::string output_dir = (argc > 3) ? argv[3] : "/Users/zhanghaining/2026code/jhcv_lib/result";

    try {
        JHDeepCore::Detector detector(model_path, "", 0);

        std::vector<std::string> supported_exts = {".png", ".jpg", ".jpeg", ".bmp"};
        std::vector<std::string> file_names;
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            for (const auto& s : supported_exts) {
                if (ext == s) {
                    file_names.push_back(entry.path().string());
                    break;
                }
            }
        }

        if (file_names.empty()) {
            std::cerr << "No images found in: " << image_dir << std::endl;
            return 1;
        }
        std::sort(file_names.begin(), file_names.end());
        ensureDirectoryExists(output_dir);

        for (const auto& file_path : file_names) {
            std::string name = file_path.substr(file_path.find_last_of("/\\") + 1);
            cv::Mat image = cv::imread(file_path);
            if (image.empty()) {
                std::cerr << "Failed to read: " << file_path << std::endl;
                continue;
            }

            std::vector<cv::Mat> images = {image};
            std::vector<JHDeepCore::DetectionResult> results;

            auto t0 = std::chrono::high_resolution_clock::now();
            detector.process(images, results);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            int num_det = results.empty() ? 0 : results[0].num_detections;
            std::cout << name << ": " << num_det << " detections, " << ms << "ms" << std::endl;

            cv::Mat result_image = image.clone();
            if (!results.empty()) {
                drawDetections(result_image, results[0]);
            }
            cv::imwrite(output_dir + "/" + name, result_image);
            std::cout << "  Saved: " << output_dir + "/" + name << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
