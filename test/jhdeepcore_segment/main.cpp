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
        JHDeepCore::Segmenter segmenter(model_path, device);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        auto result = segmenter.SegmentSingle(image);

        std::cout << "Segmentation Result:" << std::endl;
        std::cout << "  Num classes: " << result.num_classes << std::endl;
        std::cout << "  Mask size: " << result.segmentation_mask.cols << "x" << result.segmentation_mask.rows << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
