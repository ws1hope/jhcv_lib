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

// ======================== 跟踪模块 ========================

/// 跟踪距离度量类型
enum class TrackDistanceType {
    Centers,       // 中心点欧氏距离
    Rects,         // 矩形欧氏距离
    IoU,           // 交并比
};

/// 跟踪器类型
enum class TrackerType {
    Universal,     // 通用跟踪器（卡尔曼+匈牙利/LAPJV）
    ByteTrack,     // ByteTrack
};

/// 跟踪器配置
struct TrackerConfig {
    TrackerType tracker_type = TrackerType::ByteTrack;
    TrackDistanceType distance_type = TrackDistanceType::IoU;
    float distance_threshold = 0.7f;      // 匹配距离阈值
    float kalman_dt = 0.3f;               // 卡尔曼滤波时间步长
    float accel_noise = 0.2f;             // 加速度噪声
    double max_lost_time = 1.0;           // 最大丢失时间（秒）
    double max_trace_length = 2.0;        // 最大轨迹长度（秒）

    // ByteTrack 专用参数
    int bytetrack_track_buffer = 30;      // ByteTrack 轨迹缓冲帧数
    float bytetrack_track_thresh = 0.5f;  // ByteTrack 高分阈值
    float bytetrack_high_thresh = 0.5f;   // ByteTrack 第二次匹配阈值
    float bytetrack_match_thresh = 0.8f;  // ByteTrack 匹配阈值
};

/// 跟踪目标结果
struct TrackedObject {
    size_t track_id;                      // 目标唯一 ID
    cv::Rect bbox;                        // 目标位置
    int class_id = -1;                    // 目标类别
    float confidence = -1.f;              // 置信度
    std::vector<cv::Point> trajectory;    // 历史轨迹点
    bool is_stable = false;               // 轨迹是否稳定
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

class TiebiaoServicePrivate;
class TiebiaoService {
  public:
    explicit TiebiaoService(const std::string &config_path);
    ~TiebiaoService();

    TiebiaoService(const TiebiaoService &) = delete;
    TiebiaoService &operator=(const TiebiaoService &) = delete;

    const struct TiebiaoServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<TiebiaoServicePrivate> m_pHandle;
};

class DispatchServicePrivate;
class DispatchService {
  public:
    explicit DispatchService(const std::string &config_path);
    ~DispatchService();

    DispatchService(const DispatchService &) = delete;
    DispatchService &operator=(const DispatchService &) = delete;

    const struct DispatchServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<DispatchServicePrivate> m_pHandle;
};

class TrackerPrivate;
class Tracker {
  public:
    explicit Tracker(const TrackerConfig &config, float fps = 30.0f);
    ~Tracker();

    Tracker(const Tracker &) = delete;
    Tracker &operator=(const Tracker &) = delete;

    /// 输入检测框列表和当前帧，更新跟踪状态，返回跟踪结果
    void update(const std::vector<Detection> &detections,
                const cv::Mat &frame,
                std::vector<TrackedObject> &tracked_objects);

    /// 获取被移除的跟踪目标 ID
    void get_removed_ids(std::vector<size_t> &removed_ids);

  private:
    std::shared_ptr<TrackerPrivate> m_pHandle;
};

std::string GetOptimalDevice();

} // namespace JHDeepCore
