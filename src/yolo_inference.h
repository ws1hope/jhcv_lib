#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <array>

struct Detection {
    cv::Rect bbox;
    float confidence;
    int classId;
    std::string className;
    std::vector<cv::Point> mask;
    float maskConfidence;
};

struct InferenceConfig {
    std::string modelPath;
    int inputWidth = 640;
    int inputHeight = 640;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    float maskThreshold = 0.5f;
    bool useGPU = true;
    int gpuId = 0;
    std::vector<std::string> classNames;
};

class YOLOInference {
public:
    explicit YOLOInference(const InferenceConfig& config);
    ~YOLOInference();

    std::vector<Detection> detect(const cv::Mat& image);

private:
    void initSession();
    void preprocess(const cv::Mat& image, float* inputTensor);
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputs,
                                       const cv::Size& originalSize);
    std::vector<Detection> postprocessDetect(const std::vector<Ort::Value>& outputs,
                                             const cv::Size& originalSize);
    std::vector<Detection> postprocessSeg(const std::vector<Ort::Value>& outputs,
                                          const cv::Size& originalSize);
    void NMS(std::vector<Detection>& detections);
    std::vector<int> NMSIndexes(const std::vector<cv::Rect>& boxes,
                                const std::vector<float>& scores,
                                float nmsThreshold);
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
    cv::Mat getProtoMask(const float* maskData, const std::vector<float>& maskCoeffs,
                         int maskH, int maskW, const cv::Size& imageSize);

    InferenceConfig m_config;
    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<std::string> m_inputNodeNames;
    std::vector<std::string> m_outputNodeNames;
    bool m_isSegmentation;
    float m_xFactor;
    float m_yFactor;
    float m_padLeft;
    float m_padTop;
};
