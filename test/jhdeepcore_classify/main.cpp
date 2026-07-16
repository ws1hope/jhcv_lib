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

    int device_id = (device == "cpu") ? -1 : 0;

    try {
        JHDeepCore::Classifier classifier(model_path, "", device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<JHDeepCore::ClassificationResult> results;
        classifier.process(images, results);

        for (auto& result : results) {
            std::cout << "Classification Result:" << std::endl;
            std::cout << "  Class ID: " << result.class_id << std::endl;
            std::cout << "  Class Name: " << result.class_name << std::endl;
            std::cout << "  Confidence: " << result.confidence << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
