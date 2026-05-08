#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

#include "penma_rec_dll.h"

static void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --image      Path to input image (required)" << std::endl;
    std::cout << "  -l, --label      Path to label detection ONNX model (required)" << std::endl;
    std::cout << "  -z, --zifu       Path to char detection ONNX model (required)" << std::endl;
    std::cout << "  -r, --ocr_rec    Path to OCR recognition ONNX model (required)" << std::endl;
    std::cout << "  -c, --cls        Path to angle classification ONNX model (optional)" << std::endl;
    std::cout << "  -L, --ocr_label  Path to OCR recognition label file (optional)" << std::endl;
    std::cout << "  -H, --heat       Heat string for luhao matching" << std::endl;
    std::cout << "  -o, --output     Path to output image (default: output/penma_dll_result.jpg)" << std::endl;
    std::cout << "  --cpu            Use CPU inference" << std::endl;
    std::cout << "  --gpu            GPU device id (default: 0)" << std::endl;
    std::cout << "  -h, --help       Show this help message" << std::endl;
}

int main(int argc, char* argv[])
{
    std::string label_model;
    std::string zifu_model;
    std::string ocr_rec_model;
    std::string ocr_rec_label;
    std::string cls_model;
    std::string imagePath;
    std::string outputPath = "output/penma_dll_result.jpg";
    std::string heat_str;
    int use_gpu = 1;
    int gpu_id = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            imagePath = argv[++i];
        } else if ((arg == "-l" || arg == "--label") && i + 1 < argc) {
            label_model = argv[++i];
        } else if ((arg == "-z" || arg == "--zifu") && i + 1 < argc) {
            zifu_model = argv[++i];
        } else if ((arg == "-r" || arg == "--ocr_rec") && i + 1 < argc) {
            ocr_rec_model = argv[++i];
        } else if ((arg == "-c" || arg == "--cls") && i + 1 < argc) {
            cls_model = argv[++i];
        } else if ((arg == "-L" || arg == "--ocr_label") && i + 1 < argc) {
            ocr_rec_label = argv[++i];
        } else if ((arg == "-H" || arg == "--heat") && i + 1 < argc) {
            heat_str = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--cpu") {
            use_gpu = 0;
        } else if (arg == "--gpu" && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (imagePath.empty() || label_model.empty() ||
        zifu_model.empty() || ocr_rec_model.empty()) {
        std::cerr << "[ERROR] Missing required arguments." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "[INFO] Initializing penma_rec_dll..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    int ret = penma_init(
        label_model.c_str(),
        zifu_model.c_str(),
        ocr_rec_model.c_str(),
        ocr_rec_label.empty() ? nullptr : ocr_rec_label.c_str(),
        cls_model.empty() ? nullptr : cls_model.c_str(),
        use_gpu,
        gpu_id
    );

    if (ret != 0) {
        std::cerr << "[ERROR] penma_init failed with code: " << ret << std::endl;
        return 1;
    }

    auto initTime = std::chrono::high_resolution_clock::now();
    auto initMs = std::chrono::duration_cast<std::chrono::milliseconds>(initTime - startTime).count();
    std::cout << "[INFO] DLL initialization: " << initMs << " ms" << std::endl;

    // Test penma_recognize_file
    {
        std::cout << std::endl;
        std::cout << "=== Test penma_recognize_file ===" << std::endl;

        char result_buf[256] = {0};
        auto inferStart = std::chrono::high_resolution_clock::now();

        ret = penma_recognize_file(
            imagePath.c_str(),
            result_buf,
            sizeof(result_buf),
            heat_str.empty() ? nullptr : heat_str.c_str()
        );

        auto inferEnd = std::chrono::high_resolution_clock::now();
        auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        std::cout << "[INFO] Inference time: " << inferMs << " ms" << std::endl;

        if (ret == 0) {
            std::cout << "[OK] Recognize success: " << result_buf << std::endl;
        } else if (ret == 1) {
            std::cout << "[WARN] Recognize failed (not detected)" << std::endl;
        } else {
            std::cerr << "[ERROR] penma_recognize_file failed with code: " << ret << std::endl;
        }
    }

    // Test penma_save_result
    {
        std::cout << std::endl;
        std::cout << "=== Test penma_save_result ===" << std::endl;
        ret = penma_save_result(outputPath.c_str());
        if (ret == 0) {
            std::cout << "[OK] Result saved to: " << outputPath << std::endl;
        } else {
            std::cout << "[WARN] penma_save_result returned: " << ret << std::endl;
        }
    }

    penma_destroy();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    std::cout << "[INFO] Total time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - startTime).count()
              << " ms" << std::endl;

    return 0;
}
