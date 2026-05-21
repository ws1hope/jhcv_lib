#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <rec_model> <rec_label> <image_path> [device_id]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string label_path = argv[2];
    std::string image_path = argv[3];
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;

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

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
