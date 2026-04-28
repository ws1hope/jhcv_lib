#include "yolo_inference.h"
#include <iostream>
#include <iomanip>
#include <chrono>

static void printUsage(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model    Path to ONNX model file (required)" << std::endl;
    std::cout << "  -i, --image    Path to input image (required)" << std::endl;
    std::cout << "  -o, --output   Path to output image (default: output/result.jpg)" << std::endl;
    std::cout << "  -c, --conf     Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  -n, --nms      NMS threshold (default: 0.45)" << std::endl;
    std::cout << "  -W, --width    Input width (default: 640)" << std::endl;
    std::cout << "  -H, --height   Input height (default: 640)" << std::endl;
    std::cout << "  --cpu          Use CPU inference" << std::endl;
    std::cout << "  --gpu          GPU device id (default: 0)" << std::endl;
    std::cout << "  -h, --help     Show this help message" << std::endl;
}

int main(int argc, char* argv[])
{
    InferenceConfig config;
    std::string imagePath;
    std::string outputPath = "output/result.jpg";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.modelPath = argv[++i];
        } else if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            imagePath = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if ((arg == "-c" || arg == "--conf") && i + 1 < argc) {
            config.confThreshold = std::stof(argv[++i]);
        } else if ((arg == "-n" || arg == "--nms") && i + 1 < argc) {
            config.nmsThreshold = std::stof(argv[++i]);
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

        YOLOInference yolo(config);

        auto initTime = std::chrono::high_resolution_clock::now();
        auto initMs = std::chrono::duration_cast<std::chrono::milliseconds>(initTime - startTime).count();
        std::cout << "[INFO] Model initialization: " << initMs << " ms" << std::endl;

        cv::Mat dummy(config.inputHeight, config.inputWidth, CV_8UC3, cv::Scalar(0, 0, 0));
        auto warmStart = std::chrono::high_resolution_clock::now();
        yolo.detect(dummy);
        auto warmEnd = std::chrono::high_resolution_clock::now();
        auto warmMs = std::chrono::duration_cast<std::chrono::milliseconds>(warmEnd - warmStart).count();
        std::cout << "[INFO] Warmup time: " << warmMs << " ms" << std::endl;

        auto inferStart = std::chrono::high_resolution_clock::now();
        auto detections = yolo.detect(image);
        auto inferEnd = std::chrono::high_resolution_clock::now();
        auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        std::cout << "[INFO] Inference time: " << inferMs << " ms" << std::endl;
        std::cout << "[INFO] Detected " << detections.size() << " objects" << std::endl;

        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            std::cout << "  [" << i << "] " << det.className
                      << " conf=" << std::fixed << std::setprecision(3) << det.confidence
                      << " box=[" << det.bbox.x << "," << det.bbox.y
                      << "," << det.bbox.width << "," << det.bbox.height << "]"
                      << (det.mask.empty() ? "" : " +mask")
                      << std::endl;
        }

        auto visStart = std::chrono::high_resolution_clock::now();

        cv::Mat result = image.clone();
        for (const auto& det : detections) {
            cv::Scalar color((det.classId * 37) % 256,
                             (det.classId * 67) % 256,
                             (det.classId * 97) % 256);

            cv::rectangle(result, det.bbox, color, 2);

            if (!det.mask.empty()) {
                cv::Mat overlay = result.clone();
                cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{det.mask}, color);
                cv::addWeighted(overlay, 0.4, result, 0.6, 0, result);
            }

            std::stringstream label;
            label << det.className << " " << std::fixed << std::setprecision(2) << det.confidence;

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            int top = std::max(det.bbox.y, textSize.height + 5);

            cv::rectangle(result,
                          cv::Point(det.bbox.x, top - textSize.height - 5),
                          cv::Point(det.bbox.x + textSize.width, top),
                          color, -1);
            cv::putText(result, label.str(), cv::Point(det.bbox.x, top - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        }

        if (cv::imwrite(outputPath, result)) {
            std::cout << "[INFO] Result saved to: " << outputPath << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to save result to: " << outputPath << std::endl;
        }

        auto visEnd = std::chrono::high_resolution_clock::now();
        auto visMs = std::chrono::duration_cast<std::chrono::milliseconds>(visEnd - visStart).count();
        std::cout << "[INFO] Visualization time: " << visMs << " ms" << std::endl;
        std::cout << "[INFO] Total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(visEnd - startTime).count()
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
