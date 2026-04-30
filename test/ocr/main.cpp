#include <iostream>
#include <opencv2/opencv.hpp>
#include "ocr_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ocr_inference <task_mode> <image_path> [det_model.onnx] [rec_model.onnx] [rec_label.txt]" << std::endl;
        std::cout << "  task_mode: det | rec | all" << std::endl;
        std::cout << "  det - detection only (needs det_model.onnx)" << std::endl;
        std::cout << "  rec - recognition only (needs rec_model.onnx and rec_label.txt)" << std::endl;
        std::cout << "  all - detection + recognition (needs all model files)" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  ocr_inference det test.png det_model.onnx" << std::endl;
        std::cout << "  ocr_inference rec test.png rec_model.onnx rec_label.txt" << std::endl;
        std::cout << "  ocr_inference all test.png det_model.onnx rec_model.onnx rec_label.txt" << std::endl;
        return -1;
    }

    std::string task_mode_str = argv[1];
    std::string image_path = argv[2];

    OCRTaskMode task_mode;
    if (task_mode_str == "det") {
        task_mode = OCRTaskMode::DET_ONLY;
    } else if (task_mode_str == "rec") {
        task_mode = OCRTaskMode::REC_ONLY;
    } else if (task_mode_str == "all") {
        task_mode = OCRTaskMode::DET_REC;
    } else {
        std::cerr << "[ERROR] Unknown task mode: " << task_mode_str << ". Use det/rec/all." << std::endl;
        return -1;
    }

    OCRInference::Params params;
    params.task_mode = task_mode;
    params.device = "gpu";

    if (task_mode == OCRTaskMode::DET_ONLY || task_mode == OCRTaskMode::DET_REC) {
        if (argc < 4) {
            std::cerr << "[ERROR] det mode requires det_model.onnx argument" << std::endl;
            return -1;
        }
        params.det_model_path = argv[3];
    }

    if (task_mode == OCRTaskMode::REC_ONLY) {
        if (argc < 4) {
            std::cerr << "[ERROR] rec mode requires rec_model.onnx argument" << std::endl;
            return -1;
        }
        params.rec_model_path = argv[3];
        if (argc >= 5) {
            params.rec_label_path = argv[4];
        }
    }

    if (task_mode == OCRTaskMode::DET_REC) {
        if (argc < 5) {
            std::cerr << "[ERROR] all mode requires det_model.onnx and rec_model.onnx arguments" << std::endl;
            return -1;
        }
        params.rec_model_path = argv[4];
        if (argc >= 6) {
            params.rec_label_path = argv[5];
        }
    }

    std::cout << "Initializing OCR engine (" << task_mode_str << ")..." << std::endl;
    OCRInference ocr(params);

    if (task_mode == OCRTaskMode::DET_ONLY) {
        std::cout << "Processing image (detection only): " << image_path << std::endl;
        auto result = ocr.detect_only(image_path);

        std::cout << "\n=== Detection Results ===" << std::endl;
        std::cout << "Detected " << result.boxes.size() << " text regions:" << std::endl;
        for (size_t i = 0; i < result.boxes.size(); ++i) {
            const auto& box = result.boxes[i];
            std::cout << "[" << i + 1 << "] Confidence: " << box.confidence
                      << " | Points: ";
            for (const auto& pt : box.points) {
                std::cout << "(" << pt.first << "," << pt.second << ") ";
            }
            std::cout << std::endl;
        }
    } else if (task_mode == OCRTaskMode::REC_ONLY) {
        std::cout << "Processing image (recognition only): " << image_path << std::endl;
        auto result = ocr.recognize_only(image_path);

        std::cout << "\n=== Recognition Results ===" << std::endl;
        std::cout << "Text: " << result.text << " | Confidence: " << result.confidence << std::endl;
    } else {
        std::cout << "Processing image (det + rec): " << image_path << std::endl;
        auto result = ocr.predict(image_path);

        std::cout << "\n=== OCR Results ===" << std::endl;
        std::cout << "Detected " << result.boxes.size() << " text regions:" << std::endl;
        for (size_t i = 0; i < result.boxes.size(); ++i) {
            const auto& box = result.boxes[i];
            std::cout << "[" << i + 1 << "] Text: " << box.text
                      << " | Confidence: " << box.confidence << std::endl;
        }
    }

    return 0;
}
