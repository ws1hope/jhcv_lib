#include <algorithm>
#include <cctype>
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

static bool isImageFile(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const char* kImageExts[] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };
    for (const char* image_ext : kImageExts) {
        if (ext == image_ext) {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> collectImagePaths(const std::string& input_path)
{
    std::vector<std::string> paths;
    const std::filesystem::path input(input_path);

    if (std::filesystem::is_directory(input)) {
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        std::sort(paths.begin(), paths.end());
    } else if (std::filesystem::is_regular_file(input)) {
        paths.push_back(input_path);
    }

    return paths;
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

static cv::Point clampPutTextOrigin(const std::string& text,
                                    cv::Point origin,
                                    double font_scale,
                                    int thickness,
                                    const cv::Size& image_size,
                                    int margin = 4)
{
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

    int x = origin.x;
    int y = origin.y;

    const int left = x;
    const int top = y - text_size.height;
    const int right = x + text_size.width;
    const int bottom = y + baseline;

    if (left < margin) {
        x += margin - left;
    }
    if (top < margin) {
        y += margin - top;
    }
    if (right > image_size.width - margin) {
        x -= right - (image_size.width - margin);
    }
    if (bottom > image_size.height - margin) {
        y -= bottom - (image_size.height - margin);
    }

    return cv::Point(x, y);
}

static cv::Point cornerTextOrigin(const cv::Point2f& corner,
                                  const cv::Point2f& center,
                                  const std::string& text,
                                  double font_scale,
                                  int thickness,
                                  const cv::Size& image_size)
{
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

    cv::Point2f dir = center - corner;
    float norm = cv::norm(dir);
    if (norm > 1.f) {
        dir /= norm;
    } else {
        dir = cv::Point2f(1.f, 1.f);
    }

    // Bias text away from image borders when the corner sits near an edge.
    if (corner.x < image_size.width * 0.3f) {
        dir.x += 0.6f;
    } else if (corner.x > image_size.width * 0.7f) {
        dir.x -= 0.6f;
    }
    if (corner.y < image_size.height * 0.3f) {
        dir.y += 0.6f;
    } else if (corner.y > image_size.height * 0.7f) {
        dir.y -= 0.6f;
    }

    norm = cv::norm(dir);
    if (norm > 1e-3f) {
        dir /= norm;
    }

    const float offset = 22.f;
    return cv::Point(
        static_cast<int>(corner.x + dir.x * offset),
        static_cast<int>(corner.y + dir.y * offset + text_size.height * 0.5f));
}

static void drawItems(cv::Mat& image, const std::vector<JHDeepCore::SectionAngleItem>& items)
{
    const cv::Size image_size = image.size();
    constexpr double kAngleFontScale = 0.5;
    constexpr int kAngleThickness = 1;
    constexpr double kIdFontScale = 0.6;
    constexpr int kIdThickness = 2;

    for (const auto& item : items) {
        const cv::Scalar color = item.has_alert ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Point2f center(0.f, 0.f);

        for (int i = 0; i < 4; ++i) {
            center += item.corners[i];
        }
        center *= 0.25f;

        for (int i = 0; i < 4; ++i) {
            cv::line(image, item.corners[i], item.corners[(i + 1) % 4], color, 2, cv::LINE_AA);
            cv::circle(image, item.corners[i], 5, color, -1, cv::LINE_AA);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << item.angles[i] << " deg";
            const std::string angle_text = oss.str();
            const cv::Point angle_origin = clampPutTextOrigin(
                angle_text,
                cornerTextOrigin(item.corners[i], center, angle_text,
                                 kAngleFontScale, kAngleThickness, image_size),
                kAngleFontScale, kAngleThickness, image_size, 6);
            cv::putText(image, angle_text, angle_origin,
                        cv::FONT_HERSHEY_SIMPLEX, kAngleFontScale, color,
                        kAngleThickness, cv::LINE_AA);
        }

        std::ostringstream id_oss;
        id_oss << "#" << item.instance_id << (item.has_alert ? " ALERT" : " OK");
        const std::string id_text = id_oss.str();
        const cv::Point id_origin = clampPutTextOrigin(
            id_text,
            cv::Point(static_cast<int>(center.x - 20.f),
                      static_cast<int>(center.y + 5.f)),
            kIdFontScale, kIdThickness, image_size, 6);
        cv::putText(image, id_text, id_origin,
                    cv::FONT_HERSHEY_SIMPLEX, kIdFontScale, color,
                    kIdThickness, cv::LINE_AA);
    }
}

static bool processOneImage(JHDeepCore::SectionAngleChecker& checker,
                            const std::string& image_path,
                            int target_class_id)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return false;
    }

    std::cout << "\n========== " << image_path << " ==========" << std::endl;

    std::vector<JHDeepCore::SectionAngleItem> items;
    checker.process(image, items);
    printItems(items, target_class_id);

    cv::Mat result_image = image.clone();
    drawItems(result_image, items);

    const std::string output_path =
        "result/" + std::filesystem::path(image_path).filename().string();
    if (cv::imwrite(output_path, result_image)) {
        std::cout << "Result saved to: " << output_path << std::endl;
        return true;
    }

    std::cerr << "Failed to save result to: " << output_path << std::endl;
    return false;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <image_path_or_dir> [label_path] [device_id] [class_id]"
                  << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string label_path = (argc > 3) ? argv[3] : "";
    const int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    const int target_class_id = (argc > 5) ? std::stoi(argv[5]) : 1;

    try {
        const std::vector<std::string> image_paths = collectImagePaths(input_path);
        if (image_paths.empty()) {
            std::cerr << "No image files found in: " << input_path << std::endl;
            return 1;
        }

        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        JHDeepCore::SectionAngleChecker checker(
            model_path, target_class_id, kAngleToleranceDeg, label_path, device_id);

        int success_count = 0;
        for (const auto& image_path : image_paths) {
            if (processOneImage(checker, image_path, target_class_id)) {
                ++success_count;
            }
        }

        std::cout << "\nProcessed " << success_count << "/" << image_paths.size()
                  << " image(s)." << std::endl;

        return success_count == static_cast<int>(image_paths.size()) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
