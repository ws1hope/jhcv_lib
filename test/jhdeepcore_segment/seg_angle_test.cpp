#include <iostream>
#include <sstream>
#include <iomanip>
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

static constexpr float kAngleToleranceDeg = 8.0f;

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

static void printItems(const std::vector<JHDeepCore::SectionAngleItem>& items, int class_id)
{
    if (items.empty()) {
        std::cout << "[AngleCheck] Class " << class_id << " mask is empty, skip angle check"
                  << std::endl;
        return;
    }

    int ok_count = 0;
    for (const auto& item : items) {
        std::cout << "[AngleCheck] Class " << class_id
                  << " instance " << item.instance_id << std::endl;
        std::cout << "[AngleCheck] Corner angles: "
                  << std::fixed << std::setprecision(1)
                  << item.angles[0] << ", "
                  << item.angles[1] << ", "
                  << item.angles[2] << ", "
                  << item.angles[3] << std::endl;
        if (item.has_alert) {
            std::cout << "[AngleCheck] Status: ALERT" << std::endl;
        } else {
            std::cout << "[AngleCheck] Status: OK (all within 90+/-" << kAngleToleranceDeg
                      << " deg)" << std::endl;
            ++ok_count;
        }
    }

    std::cout << "[AngleCheck] Class " << class_id
              << " total instances: " << items.size()
              << ", OK: " << ok_count
              << ", ALERT: " << (items.size() - ok_count) << std::endl;
}

static void drawItems(cv::Mat& image, const std::vector<JHDeepCore::SectionAngleItem>& items)
{
    for (const auto& item : items) {
        const cv::Scalar color = item.has_alert ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Point2f center(0.f, 0.f);

        for (int i = 0; i < 4; ++i) {
            center += item.corners[i];
            cv::line(image, item.corners[i], item.corners[(i + 1) % 4], color, 2, cv::LINE_AA);
            cv::circle(image, item.corners[i], 5, color, -1, cv::LINE_AA);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << item.angles[i] << " deg";
            cv::putText(image, oss.str(),
                        cv::Point(static_cast<int>(item.corners[i].x + 8),
                                  static_cast<int>(item.corners[i].y - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        }

        center *= 0.25f;
        std::ostringstream id_oss;
        id_oss << "#" << item.instance_id << (item.has_alert ? " ALERT" : " OK");
        cv::putText(image, id_oss.str(),
                    cv::Point(static_cast<int>(center.x - 20.f),
                              static_cast<int>(center.y)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <image_path> [label_path] [device_id] [class_id]"
                  << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = (argc > 3) ? argv[3] : "";
    const int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    const int target_class_id = (argc > 5) ? std::stoi(argv[5]) : 1;

    try {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        JHDeepCore::SectionAngleChecker checker(
            model_path, target_class_id, kAngleToleranceDeg, label_path, device_id);

        std::vector<JHDeepCore::SectionAngleItem> items;
        checker.process(image, items);
        printItems(items, target_class_id);

        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        cv::Mat result_image = image.clone();
        drawItems(result_image, items);

        const std::string output_path =
            "result/" + std::filesystem::path(image_path).filename().string();
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
