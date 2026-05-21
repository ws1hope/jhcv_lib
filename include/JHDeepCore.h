#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace JHDeepCore {

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

class ClassifierPrivate;
class Classifier {
  public:
    Classifier(const std::string &model_path, const std::string &label_path = "",
               int device_id = 0, const std::string &config_path = "");
    ~Classifier();

    Classifier(const Classifier &) = delete;
    Classifier &operator=(const Classifier &) = delete;

    void process(std::vector<cv::Mat> &images, std::vector<ClassificationResult> &results);
    size_t GetBatch() const;
    size_t GetInputWidth() const;
    size_t GetInputHeight() const;

  private:
    std::shared_ptr<ClassifierPrivate> m_pHandle;
};

class DetectorPrivate;
class Detector {
  public:
    Detector(const std::string &model_path, const std::string &label_path = "",
             int device_id = 0, const std::string &config_path = "");
    ~Detector();

    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    void process(std::vector<cv::Mat> &images, std::vector<DetectionResult> &results);
    size_t GetBatch() const;
    size_t GetInputWidth() const;
    size_t GetInputHeight() const;

  private:
    std::shared_ptr<DetectorPrivate> m_pHandle;
};

class SegmenterPrivate;
class Segmenter {
  public:
    Segmenter(const std::string &model_path, const std::string &label_path = "",
              int device_id = 0, const std::string &config_path = "");
    ~Segmenter();

    Segmenter(const Segmenter &) = delete;
    Segmenter &operator=(const Segmenter &) = delete;

    void process(std::vector<cv::Mat> &images, std::vector<SegmentationResult> &results);
    size_t GetBatch() const;
    size_t GetInputWidth() const;
    size_t GetInputHeight() const;

  private:
    std::shared_ptr<SegmenterPrivate> m_pHandle;
};

class InstanceSegmenterPrivate;
class InstanceSegmenter {
  public:
    InstanceSegmenter(const std::string &model_path, const std::string &label_path = "",
                      int device_id = 0, const std::string &config_path = "");
    ~InstanceSegmenter();

    InstanceSegmenter(const InstanceSegmenter &) = delete;
    InstanceSegmenter &operator=(const InstanceSegmenter &) = delete;

    void process(std::vector<cv::Mat> &images, std::vector<InstanceSegmentationResult> &results);
    size_t GetBatch() const;
    size_t GetInputWidth() const;
    size_t GetInputHeight() const;

  private:
    std::shared_ptr<InstanceSegmenterPrivate> m_pHandle;
};

class OCRRecognizerPrivate;
class OCRRecognizer {
  public:
    OCRRecognizer(const std::string &model_path, const std::string &label_path = "",
                  int device_id = 0, const std::string &config_path = "",
                  float score_threshold = 0.5f);
    ~OCRRecognizer();

    OCRRecognizer(const OCRRecognizer &) = delete;
    OCRRecognizer &operator=(const OCRRecognizer &) = delete;

    void process(std::vector<cv::Mat> &images, std::vector<OCRResult> &results);
    size_t GetBatch() const;
    size_t GetInputWidth() const;
    size_t GetInputHeight() const;

  private:
    std::shared_ptr<OCRRecognizerPrivate> m_pHandle;
};

class OCRServicePrivate;
class OCRService {
  public:
    explicit OCRService(const std::string &config_path);
    ~OCRService();

    OCRService(const OCRService &) = delete;
    OCRService &operator=(const OCRService &) = delete;

    const struct ServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<OCRServicePrivate> m_pHandle;
};

std::string GetOptimalDevice();

} // namespace JHDeepCore
