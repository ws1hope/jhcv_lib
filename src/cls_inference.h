#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

struct ClassificationResult {
    int classId;
    float confidence;
    std::string className;
};

struct ClsConfig {
    std::string modelPath;
    std::string labelPath;
    int inputWidth = 224;
    int inputHeight = 224;
    bool useGPU = true;
    int gpuId = 0;
};

class ResNetInference {
public:
    explicit ResNetInference(const ClsConfig& config);
    ~ResNetInference();

    ClassificationResult classify(const cv::Mat& image);
    std::vector<std::pair<int, float>> classifyTopK(const cv::Mat& image, int k);

private:
    void initSession();
    void loadLabels();
    void preprocess(const cv::Mat& image, float* inputTensor);

    ClsConfig m_config;
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<std::string> m_inputNodeNames;
    std::vector<std::string> m_outputNodeNames;
    std::vector<std::string> m_labels;
};
