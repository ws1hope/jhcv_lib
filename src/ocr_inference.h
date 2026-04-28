#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct OCRDetectBox {
    std::vector<std::pair<float, float>> points;
    std::string text;
    float confidence;
};

struct OCRDetectResult {
    std::vector<OCRDetectBox> boxes;
};

class OCRInference {
public:
    struct Params {
        std::string text_detection_model_dir;
        std::string text_recognition_model_dir;
        std::string device = "cpu";
        bool use_doc_orientation_classify = false;
        bool use_doc_unwarping = false;
        bool use_textline_orientation = false;
        float text_det_thresh = 0.3f;
        float text_det_box_thresh = 0.5f;
        float text_det_unclip_ratio = 1.5f;
        float text_rec_score_thresh = 0.5f;
        int cpu_threads = 8;
        bool enable_mkldnn = true;
    };

    explicit OCRInference(const Params& params);
    ~OCRInference();

    OCRDetectResult predict(const std::string& image_path);
    OCRDetectResult predict(const cv::Mat& image);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
