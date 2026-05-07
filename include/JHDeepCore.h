#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace JHDeepCore {

enum class TaskType {
    CLASSIFICATION,
    SEGMENTATION,
    DETECTION,
    INSTANCE_SEGMENTATION,
    OCR
};

struct ClassificationResult {
    int class_id;
    std::string class_name;
    float confidence;
    std::vector<float> probabilities;
};

struct SegmentationResult {
    cv::Mat segmentation_mask;
    int num_classes;
    cv::Size image_shape;
    std::vector<std::string> class_names;
};

struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
    std::string class_name;
};

struct DetectionResult {
    std::vector<Detection> detections;
    int num_detections;
    cv::Size image_shape;
};

struct InstanceSegmentationResult {
    std::vector<Detection> detections;
    std::vector<cv::Mat> masks;
    int num_detections;
    cv::Size image_shape;
};

struct OCRBox {
    std::vector<std::pair<float, float>> points;
    std::string text;
    float confidence;
};

struct OCRResult {
    std::vector<OCRBox> boxes;
};

class Classifier {
  public:
    Classifier(const std::string &model_path, const std::string &device = "cpu");
    ~Classifier();

    Classifier(const Classifier &) = delete;
    Classifier &operator=(const Classifier &) = delete;

    ClassificationResult ClassifySingle(const cv::Mat &image);
    std::vector<ClassificationResult> ClassifyBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

class Segmenter {
  public:
    Segmenter(const std::string &model_path, const std::string &device = "cpu");
    ~Segmenter();

    Segmenter(const Segmenter &) = delete;
    Segmenter &operator=(const Segmenter &) = delete;

    SegmentationResult SegmentSingle(const cv::Mat &image);
    std::vector<SegmentationResult> SegmentBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

class Detector {
  public:
    Detector(const std::string &model_path, const std::string &device = "cpu");
    ~Detector();

    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    DetectionResult DetectSingle(const cv::Mat &image);
    std::vector<DetectionResult> DetectBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

class InstanceSegmenter {
  public:
    InstanceSegmenter(const std::string &model_path, const std::string &device = "cpu",
                      const std::vector<std::string> &class_names = {});
    ~InstanceSegmenter();

    InstanceSegmenter(const InstanceSegmenter &) = delete;
    InstanceSegmenter &operator=(const InstanceSegmenter &) = delete;

    InstanceSegmentationResult SegmentSingle(const cv::Mat &image);
    std::vector<InstanceSegmentationResult> SegmentBatch(const std::vector<cv::Mat> &images);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

class OCRRecognizer {
  public:
    struct Params {
        std::string rec_model_path;
        std::string rec_label_path;
        std::string device = "cpu";
        float text_rec_score_thresh = 0.5f;
        bool useGPU = true;
        int gpuId = 0;
    };

    explicit OCRRecognizer(const Params &params);
    ~OCRRecognizer();

    OCRRecognizer(const OCRRecognizer &) = delete;
    OCRRecognizer &operator=(const OCRRecognizer &) = delete;

    OCRResult Recognize(const cv::Mat &text_image);
    OCRResult Recognize(const std::string &image_path);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

std::string GetOptimalDevice();

} // namespace JHDeepCore
