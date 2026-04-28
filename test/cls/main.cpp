#include "cls_inference.h"
#include <iostream>
#include <iomanip>
#include <chrono>

static void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model    Path to ONNX model file (required)" << std::endl;
    std::cout << "  -i, --image    Path to input image (required)" << std::endl;
    std::cout << "  -l, --labels   Path to label file (one class per line)" << std::endl;
    std::cout << "  -k, --topk     Show top-K results (default: 5)" << std::endl;
    std::cout << "  -W, --width    Input width (default: 224)" << std::endl;
    std::cout << "  -H, --height   Input height (default: 224)" << std::endl;
    std::cout << "  --cpu          Use CPU inference" << std::endl;
    std::cout << "  --gpu          GPU device id (default: 0)" << std::endl;
    std::cout << "  -h, --help     Show this help message" << std::endl;
}

int main(int argc, char* argv[])
{
    ClsConfig config;
    std::string imagePath;
    int topK = 5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.modelPath = argv[++i];
        } else if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            imagePath = argv[++i];
        } else if ((arg == "-l" || arg == "--labels") && i + 1 < argc) {
            config.labelPath = argv[++i];
        } else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) {
            topK = std::stoi(argv[++i]);
        } else if ((arg == "-W" || arg == "--width") && i + 1 < argc) {
            config.inputWidth = std::stoi(argv[++i]);
        } else if ((arg == "-H" || arg == "--height") && i + 1 < argc) {
            config.inputHeight = std::stoi(argv[++i]);
        } else if (arg == "--cpu") {
            config.useGPU = false;
        } else if (arg == "--gpu" && i + 1 < argc) {
            config.gpuId = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (config.modelPath.empty() || imagePath.empty()) {
        std::cerr << "[ERROR] Model path and image path are required." << std::endl;
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

        ResNetInference resnet(config);

        auto initTime = std::chrono::high_resolution_clock::now();
        auto initMs = std::chrono::duration_cast<std::chrono::milliseconds>(initTime - startTime).count();
        std::cout << "[INFO] Model initialization: " << initMs << " ms" << std::endl;

        cv::Mat dummy(config.inputHeight, config.inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        auto warmStart = std::chrono::high_resolution_clock::now();
        resnet.classify(dummy);
        auto warmEnd = std::chrono::high_resolution_clock::now();
        auto warmMs = std::chrono::duration_cast<std::chrono::milliseconds>(warmEnd - warmStart).count();
        std::cout << "[INFO] Warmup time: " << warmMs << " ms" << std::endl;

        auto inferStart = std::chrono::high_resolution_clock::now();
        auto topResults = resnet.classifyTopK(image, topK);
        auto inferEnd = std::chrono::high_resolution_clock::now();
        auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        std::cout << "[INFO] Inference time: " << inferMs << " ms" << std::endl;

        std::cout << "[INFO] Top-" << topK << " predictions:" << std::endl;
        for (int i = 0; i < static_cast<int>(topResults.size()); i++) {
            int classId = topResults[i].first;
            float score = topResults[i].second;
            std::string name = "class_" + std::to_string(classId);

            std::cout << "  [" << i << "] " << std::fixed << std::setprecision(4) << score
                      << " - " << name << " (id=" << classId << ")" << std::endl;
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        std::cout << "[INFO] Total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - startTime).count()
                  << " ms" << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}