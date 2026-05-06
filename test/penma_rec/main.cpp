#include "penma_rec_inference.h"
#include <iostream>
#include <chrono>

static void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --image      Path to input image (required)" << std::endl;
    std::cout << "  -l, --label      Path to label detection ONNX model (required)" << std::endl;
    std::cout << "  -z, --zifu       Path to char detection ONNX model (required)" << std::endl;
    std::cout << "  -r, --ocr_rec    Path to OCR recognition ONNX model (required)" << std::endl;
    std::cout << "  -c, --cls        Path to angle classification ONNX model (optional)" << std::endl;
    std::cout << "  -L, --ocr_label  Path to OCR recognition label file" << std::endl;
    std::cout << "  -H, --heat       Heat string for luhao matching" << std::endl;
    std::cout << "  -o, --output     Path to output image (default: output/penma_result.jpg)" << std::endl;
    std::cout << "  --cpu            Use CPU inference" << std::endl;
    std::cout << "  --gpu            GPU device id (default: 0)" << std::endl;
    std::cout << "  -h, --help       Show this help message" << std::endl;
}

int main(int argc, char* argv[])
{
    PenmaRecParams params;
    std::string imagePath;
    std::string outputPath = "output/penma_result.jpg";
    std::string heat_str;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            imagePath = argv[++i];
        } else if ((arg == "-l" || arg == "--label") && i + 1 < argc) {
            params.label_model_path = argv[++i];
        } else if ((arg == "-z" || arg == "--zifu") && i + 1 < argc) {
            params.zifu_model_path = argv[++i];
        } else if ((arg == "-r" || arg == "--ocr_rec") && i + 1 < argc) {
            params.ocr_rec_model_path = argv[++i];
        } else if ((arg == "-c" || arg == "--cls") && i + 1 < argc) {
            params.cls_model_path = argv[++i];
        } else if ((arg == "-L" || arg == "--ocr_label") && i + 1 < argc) {
            params.ocr_rec_label_path = argv[++i];
        } else if ((arg == "-H" || arg == "--heat") && i + 1 < argc) {
            heat_str = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--cpu") {
            params.useGPU = false;
        } else if (arg == "--gpu" && i + 1 < argc) {
            params.gpuId = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (imagePath.empty() || params.label_model_path.empty() ||
        params.zifu_model_path.empty() || params.ocr_rec_model_path.empty()) {
        std::cerr << "[ERROR] Missing required arguments." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    std::cout << "[INFO] Image size: " << image.cols << "x" << image.rows << std::endl;

    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        PenmaRecInference rec(params);

        auto initTime = std::chrono::high_resolution_clock::now();
        auto initMs = std::chrono::duration_cast<std::chrono::milliseconds>(initTime - startTime).count();
        std::cout << "[INFO] Model initialization: " << initMs << " ms" << std::endl;

        auto inferStart = std::chrono::high_resolution_clock::now();
        auto result = rec.recognize(image, heat_str);
        auto inferEnd = std::chrono::high_resolution_clock::now();
        auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        std::cout << "[INFO] Inference time: " << inferMs << " ms" << std::endl;

        std::cout << std::endl;
        std::cout << "=== PenmaRec Results ===" << std::endl;
        std::cout << "Success: " << (result.success ? "YES" : "NO") << std::endl;
        std::cout << "Labels detected: " << result.labels.size() << std::endl;

        for (size_t i = 0; i < result.labels.size(); i++) {
            std::cout << "  Label[" << i << "] class=" << result.labels[i].class_id
                      << " bbox=[" << result.labels[i].bbox.x << "," << result.labels[i].bbox.y
                      << "," << result.labels[i].bbox.width << "," << result.labels[i].bbox.height << "]"
                      << std::endl;
        }

        std::cout << "Characters recognized: " << result.characters.size() << std::endl;
        for (size_t i = 0; i < result.characters.size(); i++) {
            std::cout << "  Char[" << i << "] text=\"" << result.characters[i].text
                      << "\" cls_label=" << result.characters[i].cls_label
                      << " center=(" << result.characters[i].center_x
                      << "," << result.characters[i].center_y << ")" << std::endl;
        }

        std::cout << "OCR result: " << result.ocr_result << std::endl;

        if (!result.rotated_image.empty()) {
            if (cv::imwrite(outputPath, result.rotated_image)) {
                std::cout << "[INFO] Rotated result saved to: " << outputPath << std::endl;
            }
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::cout << "[INFO] Total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - startTime).count()
                  << " ms" << std::endl;

        return result.success ? 0 : 2;

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
