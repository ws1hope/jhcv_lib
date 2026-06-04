#include "cls_inference.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <thread>

ResNetInference::ResNetInference(const ClsConfig& config)
    : m_config(config), m_env(ORT_LOGGING_LEVEL_WARNING, "resnet")
{
    loadLabels();
    initSession();
}

ResNetInference::~ResNetInference() = default;

void ResNetInference::loadLabels()
{
    if (m_config.labelPath.empty()) {
        std::cout << "[WARN] No label file provided, using numeric class IDs" << std::endl;
        return;
    }

    std::ifstream file(m_config.labelPath);
    if (!file.is_open()) {
        std::cerr << "[WARN] Failed to open label file: " << m_config.labelPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            m_labels.push_back(line);
        }
    }
    std::cout << "[INFO] Loaded " << m_labels.size() << " class labels" << std::endl;
}

void ResNetInference::initSession()
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

#ifdef _WIN32
    std::wstring wideModelPath(m_config.modelPath.begin(), m_config.modelPath.end());
    m_session = std::make_unique<Ort::Session>(m_env, wideModelPath.c_str(), sessionOptions);
#else
    m_session = std::make_unique<Ort::Session>(m_env, m_config.modelPath.c_str(), sessionOptions);
#endif

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
    }

    m_inputNames.clear();
    for (auto& n : m_inputNodeNames) {
        m_inputNames.push_back(n.c_str());
    }
    m_outputNames.clear();
    for (auto& n : m_outputNodeNames) {
        m_outputNames.push_back(n.c_str());
    }

    std::cout << "[INFO] Model: " << m_config.modelPath << std::endl;
    std::cout << "[INFO] Input: " << m_inputNodeNames[0]
              << " [" << m_config.inputWidth << "x" << m_config.inputHeight << "]" << std::endl;
    std::cout << "[INFO] Output: " << m_outputNodeNames[0] << std::endl;
}

void ResNetInference::preprocess(const cv::Mat& image, float* inputTensor)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(m_config.inputWidth, m_config.inputHeight), 0, 0, cv::INTER_LINEAR);

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0f / 255.0f);

    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);

    int channelSize = m_config.inputWidth * m_config.inputHeight;
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];
        std::memcpy(inputTensor + c * channelSize, channels[c].data, channelSize * sizeof(float));
    }
}

ClassificationResult ResNetInference::classify(const cv::Mat& image)
{
    auto results = classifyTopK(image, 1);
    if (results.empty()) {
        return {-1, 0.0f, "unknown"};
    }

    ClassificationResult result;
    result.classId = results[0].first;
    result.confidence = results[0].second;

    if (result.classId >= 0 && result.classId < static_cast<int>(m_labels.size())) {
        result.className = m_labels[result.classId];
    } else {
        result.className = "class_" + std::to_string(result.classId);
    }

    return result;
}

std::vector<std::pair<int, float>> ResNetInference::classifyTopK(const cv::Mat& image, int k)
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

    const float* outputData = outputs[0].GetTensorData<float>();
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int numClasses = static_cast<int>(outputShape[1]);

    std::vector<std::pair<int, float>> scores;
    scores.reserve(numClasses);
    for (int i = 0; i < numClasses; i++) {
        scores.emplace_back(i, outputData[i]);
    }

    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    scores.resize(k);
    return scores;
}