#include <iostream>
#include <opencv2/opencv.hpp>
#include "ocr_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ocr_inference <image_path> <det_model_dir> <rec_model_dir>" << std::endl;
        std::cout << "Example: ocr_inference test.png G:/models/det G:/models/rec" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    std::string det_model_dir = argv[2];
    std::string rec_model_dir = argv[3];

    OCRInference::Params params;
    params.text_detection_model_dir = det_model_dir;
    params.text_recognition_model_dir = rec_model_dir;
    params.cpu_threads = 8;
    params.enable_mkldnn = true;

    std::cout << "Initializing OCR engine..." << std::endl;
    OCRInference ocr(params);

    std::cout << "Processing image: " << image_path << std::endl;
    auto result = ocr.predict(image_path);

    std::cout << "\n=== OCR Results ===" << std::endl;
    std::cout << "Detected " << result.boxes.size() << " text regions:" << std::endl;
    
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const auto& box = result.boxes[i];
        std::cout << "[" << i + 1 << "] Text: " << box.text 
                  << " | Confidence: " << box.confidence << std::endl;
    }

    return 0;
}