#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

enum class OCRTaskMode {
    DET_ONLY,
    REC_ONLY,
    DET_REC
};

struct OCRDetectBox {
    std::vector<std::pair<float, float>> points;
    std::string text;
    float confidence;
};

struct OCRDetectResult {
    std::vector<OCRDetectBox> boxes;
};

struct OCRRecResult {
    std::string text;
    float confidence;
};

class OCRInference {
public:
    struct Params {
        std::string det_model_path;
        std::string rec_model_path;
        std::string rec_label_path;
        std::string device = "cpu";
        OCRTaskMode task_mode = OCRTaskMode::DET_REC;
        int det_img_h = 960;
        int det_img_w = 960;
        float text_det_thresh = 0.3f;
        float text_det_box_thresh = 0.5f;
        float text_det_unclip_ratio = 1.5f;
        float text_rec_score_thresh = 0.5f;
        bool useGPU = true;
        int gpuId = 0;
    };

    explicit OCRInference(const Params& params);
    ~OCRInference();

    OCRDetectResult predict(const std::string& image_path);
    OCRDetectResult predict(const cv::Mat& image);

    OCRDetectResult detect_only(const std::string& image_path);
    OCRDetectResult detect_only(const cv::Mat& image);

    OCRRecResult recognize_only(const cv::Mat& text_image);
    OCRRecResult recognize_only(const std::string& image_path);

private:
    void initDetSession();
    void initRecSession();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
