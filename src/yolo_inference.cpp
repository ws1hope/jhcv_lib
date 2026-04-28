#include "yolo_inference.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <cmath>
#include <codecvt>
#include <locale>

static std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

YOLOInference::YOLOInference(const InferenceConfig& config)
    : m_config(config), m_env(ORT_LOGGING_LEVEL_WARNING, "yolo"), m_isSegmentation(false),
      m_xFactor(1.0f), m_yFactor(1.0f)
{
    if (m_config.classNames.empty()) {
        m_config.classNames = COCO_CLASSES;
    }
    initSession();
}

YOLOInference::~YOLOInference() = default;

void YOLOInference::initSession()
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    if (m_config.useGPU) {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = m_config.gpuId;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "[INFO] Using CUDA GPU (device " << m_config.gpuId << ")" << std::endl;
    } else {
        std::cout << "[INFO] Using CPU" << std::endl;
    }
#else
    std::cout << "[INFO] Using CPU (CUDA not enabled)" << std::endl;
#endif

    std::wstring wideModelPath(m_config.modelPath.begin(), m_config.modelPath.end());
    m_session = std::make_unique<Ort::Session>(m_env, wideModelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = m_session->GetInputCount();
    for (size_t i = 0; i < numInputNodes; i++) {
        auto name = m_session->GetInputNameAllocated(i, allocator);
        m_inputNodeNames.emplace_back(name.get());
    }

    size_t numOutputNodes = m_session->GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; i++) {
        auto name = m_session->GetOutputNameAllocated(i, allocator);
        m_outputNodeNames.emplace_back(name.get());
        std::cout << "[DEBUG] Output[" << i << "] name: '" << m_outputNodeNames.back() << "'" << std::endl;
    }

    m_inputNames.clear();
    for (auto& n : m_inputNodeNames) {
        m_inputNames.push_back(n.c_str());
    }
    m_outputNames.clear();
    for (auto& n : m_outputNodeNames) {
        m_outputNames.push_back(n.c_str());
    }

    m_isSegmentation = (numOutputNodes >= 2);
    std::cout << "[INFO] Model: " << m_config.modelPath << std::endl;
    std::cout << "[INFO] Mode: " << (m_isSegmentation ? "Detection + Segmentation" : "Detection Only") << std::endl;
    std::cout << "[INFO] Input: " << m_inputNodeNames[0]
              << " [" << m_config.inputWidth << "x" << m_config.inputHeight << "]" << std::endl;
    std::cout << "[INFO] Outputs: " << numOutputNodes << std::endl;
}

void YOLOInference::preprocess(const cv::Mat& image, float* inputTensor)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    int imgW = rgb.cols;
    int imgH = rgb.rows;

    float scale = std::min(static_cast<float>(m_config.inputWidth) / imgW,
                           static_cast<float>(m_config.inputHeight) / imgH);
    int newW = static_cast<int>(imgW * scale);
    int newH = static_cast<int>(imgH * scale);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    int top = (m_config.inputHeight - newH) / 2;
    int left = (m_config.inputWidth - newW) / 2;

    cv::Mat padded(m_config.inputHeight, m_config.inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, newW, newH)));

    cv::Mat floatImg;
    padded.convertTo(floatImg, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);

    int channelSize = m_config.inputWidth * m_config.inputHeight;
    for (int c = 0; c < 3; c++) {
        std::memcpy(inputTensor + c * channelSize, channels[c].data, channelSize * sizeof(float));
    }

    m_xFactor = scale;
    m_yFactor = scale;
    m_padLeft = static_cast<float>(left);
    m_padTop = static_cast<float>(top);
}

std::vector<Detection> YOLOInference::detect(const cv::Mat& image)
{
    std::vector<float> inputData(3 * m_config.inputWidth * m_config.inputHeight);
    preprocess(image, inputData.data());

    std::array<int64_t, 4> inputShape = {1, 3, m_config.inputHeight, m_config.inputWidth};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

    auto outputs = m_session->Run(Ort::RunOptions{nullptr},
                                  m_inputNames.data(), &inputTensor, 1,
                                  m_outputNames.data(), m_outputNames.size());
                    
    return postprocess(outputs, cv::Size(image.cols, image.rows));
}

std::vector<Detection> YOLOInference::postprocess(const std::vector<Ort::Value>& outputs,
                                                   const cv::Size& originalSize)
{
    if (m_isSegmentation) {
        return postprocessSeg(outputs, originalSize);
    }
    return postprocessDetect(outputs, originalSize);
}

std::vector<Detection> YOLOInference::postprocessDetect(const std::vector<Ort::Value>& outputs,
                                                        const cv::Size& originalSize)
{
    std::vector<Detection> detections;
    auto& output = outputs[0];
    auto typeInfo = output.GetTensorTypeAndShapeInfo();
    auto outputShape = typeInfo.GetShape();

    int numDims = static_cast<int>(outputShape.size());
    int numDetections;
    int numClasses;

    if (numDims == 3) {
        numDetections = static_cast<int>(outputShape[2]);
        numClasses = static_cast<int>(outputShape[1]) - 5;
    } else {
        numDetections = static_cast<int>(outputShape[1]);
        numClasses = static_cast<int>(outputShape[2]) - 5;
    }

    const float* data = output.GetTensorData<float>();

    for (int i = 0; i < numDetections; i++) {
        const float* row = (numDims == 3) ? data + i : data + i * (numClasses + 5);
        if (numDims == 3) {
            row = data + (numClasses + 5) * i;
        }

        float objConf = row[4];
        if (objConf < m_config.confThreshold) continue;

        int bestClass = 0;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; c++) {
            float score = row[5 + c];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        float confidence = objConf * bestScore;
        if (confidence < m_config.confThreshold) continue;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        int x1 = static_cast<int>((cx - w / 2.0f) * m_xFactor);
        int y1 = static_cast<int>((cy - h / 2.0f) * m_yFactor);
        int x2 = static_cast<int>((cx + w / 2.0f) * m_xFactor);
        int y2 = static_cast<int>((cy + h / 2.0f) * m_yFactor);

        x1 = std::max(0, std::min(x1, originalSize.width));
        y1 = std::max(0, std::min(y1, originalSize.height));
        x2 = std::max(0, std::min(x2, originalSize.width));
        y2 = std::max(0, std::min(y2, originalSize.height));

        Detection det;
        det.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        det.confidence = confidence;
        det.classId = bestClass;
        det.maskConfidence = 0.0f;
        if (bestClass < static_cast<int>(m_config.classNames.size())) {
            det.className = m_config.classNames[bestClass];
        } else {
            det.className = "class_" + std::to_string(bestClass);
        }
        detections.push_back(det);
    }

    NMS(detections);
    return detections;
}

std::vector<Detection> YOLOInference::postprocessSeg(const std::vector<Ort::Value>& outputs,
                                                     const cv::Size& originalSize)
{
    std::vector<Detection> detections;

    auto& output0 = outputs[0];
    auto typeInfo0 = output0.GetTensorTypeAndShapeInfo();
    auto shape0 = typeInfo0.GetShape();

    std::cout << "[DEBUG] output0 shape: ";
    for (auto s : shape0) std::cout << s << " ";
    std::cout << std::endl;

    int numDims = static_cast<int>(shape0.size());
    const float* data0 = output0.GetTensorTypeAndShapeInfo().GetShape().size() > 0
        ? output0.GetTensorData<float>() : nullptr;

    int numDetections;
    int dataLen;

    if (numDims == 3) {
        dataLen = static_cast<int>(shape0[1]);
        numDetections = static_cast<int>(shape0[2]);
    } else {
        dataLen = static_cast<int>(shape0[2]);
        numDetections = static_cast<int>(shape0[1]);
    }

    int numClasses = dataLen - 4 - 32;

    auto& output1 = outputs[1];
    auto typeInfo1 = output1.GetTensorTypeAndShapeInfo();
    auto shape1 = typeInfo1.GetShape();
    const float* protoData = output1.GetTensorData<float>();

    int maskH = static_cast<int>(shape1[2]);
    int maskW = static_cast<int>(shape1[3]);

    std::vector<cv::Rect> boxes;
    std::vector<float> confScores;
    std::vector<int> classIds;
    std::vector<std::vector<float>> maskCoeffsList;

    for (int i = 0; i < numDetections; i++) {
        int bestClass = 0;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; c++) {
            float score = data0[(4 + c) * numDetections + i];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestScore < m_config.confThreshold) continue;

        float cx = data0[0 * numDetections + i];
        float cy = data0[1 * numDetections + i];
        float w = data0[2 * numDetections + i] / m_xFactor;
        float h = data0[3 * numDetections + i] / m_yFactor;
        int left = std::max(int((cx - m_padLeft) / m_xFactor - 0.5f * w + 0.5f), 0);
        int top = std::max(int((cy - m_padTop) / m_yFactor - 0.5f * h + 0.5f), 0);

        boxes.emplace_back(left, top, int(w + 0.5f), int(h + 0.5f));
        confScores.push_back(bestScore);
        classIds.push_back(bestClass);

        std::vector<float> coeffs(32);
        for (int m = 0; m < 32; m++) {
            coeffs[m] = data0[(4 + numClasses + m) * numDetections + i];
        }
        maskCoeffsList.push_back(std::move(coeffs));
    }

    auto nmsIdxs = NMSIndexes(boxes, confScores, m_config.nmsThreshold);

    for (int idx : nmsIdxs) {
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, originalSize.width, originalSize.height);

        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confScores[idx];
        det.classId = classIds[idx];
        det.maskConfidence = 0.0f;
        if (classIds[idx] < static_cast<int>(m_config.classNames.size())) {
            det.className = m_config.classNames[classIds[idx]];
        } else {
            det.className = "class_" + std::to_string(classIds[idx]);
        }

        cv::Mat mask = getProtoMask(protoData, maskCoeffsList[idx], maskH, maskW, originalSize);

        cv::Mat maskRoi = mask(det.bbox);
        cv::Mat binaryMask;
        cv::threshold(maskRoi, binaryMask, m_config.maskThreshold, 255, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            size_t maxIdx = 0;
            double maxArea = cv::contourArea(contours[0]);
            for (size_t c = 1; c < contours.size(); c++) {
                double area = cv::contourArea(contours[c]);
                if (area > maxArea) {
                    maxArea = area;
                    maxIdx = c;
                }
            }
            det.mask = contours[maxIdx];
            for (auto& pt : det.mask) {
                pt.x += det.bbox.x;
                pt.y += det.bbox.y;
            }
        }

        detections.push_back(det);
    }

    NMS(detections);
    return detections;
}

cv::Mat YOLOInference::getProtoMask(const float* maskData, const std::vector<float>& maskCoeffs,
                                    int maskH, int maskW, const cv::Size& imageSize)
{
    cv::Mat protos(32, maskH * maskW, CV_32F, const_cast<float*>(maskData));
    cv::Mat coeffs(1, 32, CV_32F, const_cast<float*>(maskCoeffs.data()));
    cv::Mat matmul = coeffs * protos;
    matmul = matmul.reshape(1, maskH);

    cv::Mat sigmoidMask(maskH, maskW, CV_32F);
    float* src = matmul.ptr<float>();
    float* dst = sigmoidMask.ptr<float>();
    int total = maskH * maskW;
    for (int i = 0; i < total; i++) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }

    cv::resize(sigmoidMask, sigmoidMask, imageSize, 0, 0, cv::INTER_LINEAR);
    return sigmoidMask;
}

std::vector<int> YOLOInference::NMSIndexes(const std::vector<cv::Rect>& boxes,
                                const std::vector<float>& scores,
                                float nmsThreshold)
{
    std::vector<int> indices;
    if (boxes.empty()) return indices;

    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(scores.size(), false);

    for (size_t i = 0; i < order.size(); i++) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        indices.push_back(idx);
        for (size_t j = i + 1; j < order.size(); j++) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            float iou = static_cast<float>((boxes[idx] & boxes[jdx]).area()) /
                        static_cast<float>((boxes[idx] | boxes[jdx]).area());
            if (iou > nmsThreshold) {
                suppressed[jdx] = true;
            }
        }
    }
    return indices;
}

void YOLOInference::NMS(std::vector<Detection>& detections)
{
    if (detections.empty()) return;

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;

            float iou = static_cast<float>((detections[i].bbox & detections[j].bbox).area()) /
                        static_cast<float>((detections[i].bbox | detections[j].bbox).area());
            if (iou > m_config.nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                       [&](const Detection& d) { return suppressed[&d - &detections[0]]; }),
        detections.end());
}

void YOLOInference::drawDetections(cv::Mat& image, const std::vector<Detection>& detections)
{
    for (const auto& det : detections) {
        cv::Scalar color = cv::Scalar(
            (det.classId * 37) % 256, (det.classId * 67) % 256, (det.classId * 97) % 256);

        cv::rectangle(image, det.bbox, color, 2);

        if (!det.mask.empty()) {
            cv::Mat overlay = image.clone();
            cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{det.mask}, color);
            cv::addWeighted(overlay, 0.4, image, 0.6, 0, image);
        }

        std::stringstream label;
        label << det.className << " " << std::fixed << std::setprecision(2) << det.confidence;

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int top = std::max(det.bbox.y, textSize.height + 5);

        cv::rectangle(image,
                      cv::Point(det.bbox.x, top - textSize.height - 5),
                      cv::Point(det.bbox.x + textSize.width, top),
                      color, -1);
        cv::putText(image, label.str(), cv::Point(det.bbox.x, top - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
