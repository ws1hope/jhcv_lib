#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [device]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string device = (argc > 3) ? argv[3] : "cpu";

    try {
        JHDeepCore::Detector detector(model_path, device);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        auto result = detector.DetectSingle(image);

        std::cout << "Detection Result:" << std::endl;
        std::cout << "  Num detections: " << result.num_detections << std::endl;
        for (const auto& det : result.detections) {
            std::cout << "  [" << det.class_name << "] conf=" << det.confidence
                      << " bbox=(" << det.bbox.x << "," << det.bbox.y << ","
                      << det.bbox.width << "," << det.bbox.height << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
