#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <rec_model> <rec_label> <image_path> [device]" << std::endl;
        return 1;
    }

    JHDeepCore::OCRRecognizer::Params params;
    params.rec_model_path = argv[1];
    params.rec_label_path = argv[2];
    std::string image_path = argv[3];
    params.device = (argc > 4) ? argv[4] : "cpu";

    try {
        JHDeepCore::OCRRecognizer ocr(params);

        auto result = ocr.Recognize(image_path);

        std::cout << "OCR Result:" << std::endl;
        for (const auto& box : result.boxes) {
            std::cout << "  Text: \"" << box.text << "\" conf=" << box.confidence << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
