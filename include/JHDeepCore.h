#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>
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
             int device_id = 0, const std::string &config_path = "",
             float conf_threshold = 0.25f, float iou_threshold = 0.45f);
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
                      int device_id = 0, const std::string &config_path = "",
                      float conf_threshold = 0.25f, float iou_threshold = 0.45f);
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

class ZbhcServicePrivate;
class ZbhcService {
  public:
    explicit ZbhcService(const std::string &config_path);
    ~ZbhcService();

    ZbhcService(const ZbhcService &) = delete;
    ZbhcService &operator=(const ZbhcService &) = delete;

    const struct ZbhcServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<ZbhcServicePrivate> m_pHandle;
};

class LuqianServicePrivate;
class LuqianService {
  public:
    explicit LuqianService(const std::string &config_path);
    ~LuqianService();

    LuqianService(const LuqianService &) = delete;
    LuqianService &operator=(const LuqianService &) = delete;

    const struct LuqianServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<LuqianServicePrivate> m_pHandle;
};

class XintiangangServicePrivate;
class XintiangangService {
  public:
    explicit XintiangangService(const std::string &config_path);
    ~XintiangangService();

    XintiangangService(const XintiangangService &) = delete;
    XintiangangService &operator=(const XintiangangService &) = delete;

    const struct XintiangangServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<XintiangangServicePrivate> m_pHandle;
};

class HuaxinServicePrivate;
class HuaxinService {
  public:
    explicit HuaxinService(const std::string &config_path);
    ~HuaxinService();

    HuaxinService(const HuaxinService &) = delete;
    HuaxinService &operator=(const HuaxinService &) = delete;

    const struct HuaxinServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     const std::string &station_id);

  private:
    std::shared_ptr<HuaxinServicePrivate> m_pHandle;
};

class GuokuacheServicePrivate;
class GuokuacheService {
  public:
    explicit GuokuacheService(const std::string &config_path);
    ~GuokuacheService();

    GuokuacheService(const GuokuacheService &) = delete;
    GuokuacheService &operator=(const GuokuacheService &) = delete;

    const struct GuokuacheServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<GuokuacheServicePrivate> m_pHandle;
};

class FujianServicePrivate;
class FujianService {
  public:
    explicit FujianService(const std::string &config_path);
    ~FujianService();

    FujianService(const FujianService &) = delete;
    FujianService &operator=(const FujianService &) = delete;

    const struct FujianServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<FujianServicePrivate> m_pHandle;
};

struct SectionAngleItem {
    int instance_id = 0;
    cv::Point2f corners[4];
    float angles[4] = {0.f, 0.f, 0.f, 0.f};
    bool has_alert = false;
};

class SectionAngleCheckerPrivate;
class SectionAngleChecker {
  public:
    SectionAngleChecker(const std::string &model_path,
                        int target_class_id = 1,
                        float angle_tolerance_deg = 8.0f,
                        const std::string &label_path = "",
                        int device_id = 0,
                        const std::string &config_path = "");
    ~SectionAngleChecker();

    SectionAngleChecker(const SectionAngleChecker &) = delete;
    SectionAngleChecker &operator=(const SectionAngleChecker &) = delete;

    void process(const cv::Mat &image, std::vector<SectionAngleItem> &results);

  private:
    std::shared_ptr<SectionAngleCheckerPrivate> m_pHandle;
};

class ZbsltjServicePrivate;
class ZbsltjService {
  public:
    explicit ZbsltjService(const std::string &config_path);
    ~ZbsltjService();

    ZbsltjService(const ZbsltjService &) = delete;
    ZbsltjService &operator=(const ZbsltjService &) = delete;

    const struct ZbsltjServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int camera_id);

  private:
    std::shared_ptr<ZbsltjServicePrivate> m_pHandle;
};

class GxJingzhengServicePrivate;
class GxJingzhengService {
  public:
    explicit GxJingzhengService(const std::string &config_path);
    ~GxJingzhengService();

    GxJingzhengService(const GxJingzhengService &) = delete;
    GxJingzhengService &operator=(const GxJingzhengService &) = delete;

    const struct GxJingzhengServerConfig &config() const;
    std::string handleRequest(const std::string &req_body);
    int runLocalTest(const std::string &image_path,
                     const std::string &heat_number,
                     int station_id);

  private:
    std::shared_ptr<GxJingzhengServicePrivate> m_pHandle;
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

// ======================== Cross-camera tracking ========================

using CameraId = size_t;
using TargetId = uint64_t;

/// A calibration pair mapping one camera pixel to the shared map plane.
struct CalibrationPointPair {
    cv::Point2f image_point;
    cv::Point2f map_point;
};

/// Configuration for one camera channel.
struct CrossCameraChannelConfig {
    CameraId camera_id = 0;
    float tracker_fps = 30.0f;
    std::vector<CalibrationPointPair> calibration_points;
};

/// An undirected adjacency relation between two camera channels.
struct CrossCameraLinkConfig {
    CameraId camera_a_id = 0;
    CameraId camera_b_id = 0;
    float max_distance = 200.0f;
};

/// Minimal configuration for the first cross-camera tracking demo.
struct CrossCameraTrackerConfig {
    TrackerConfig tracker_config;
    std::vector<CrossCameraChannelConfig> channels;
    std::vector<CrossCameraLinkConfig> links;
    bool enable_log = true;
    std::string log_directory = "logs";
};

/// Input for one camera in a synchronous update batch.
struct CrossCameraFrameInput {
    CameraId camera_id = 0;
    cv::Mat frame;
    std::vector<Detection> detections;
};

/// One local track augmented with a cross-camera target ID.
struct CrossCameraTrackedObject {
    CameraId camera_id = 0;
    TargetId target_id = 0;
    TrackedObject local_track;
    cv::Point2f mapped_point;
};

/// One unique, EMA-smoothed map position for a global target.
struct CrossCameraGlobalTarget {
    TargetId target_id = 0;
    CameraId camera_id = 0;
    size_t local_track_id = 0;
    /// Raw mapped point from the selected local track.
    cv::Point2f raw_mapped_point;
    /// EMA-smoothed point intended for direct visualization.
    cv::Point2f smoothed_mapped_point;
};

class CrossCameraTrackerPrivate;
class CrossCameraTracker {
  public:
    explicit CrossCameraTracker(const CrossCameraTrackerConfig &config);
    ~CrossCameraTracker();

    CrossCameraTracker(const CrossCameraTracker &) = delete;
    CrossCameraTracker &operator=(const CrossCameraTracker &) = delete;

    /// Update all configured cameras and associate tracks across adjacent cameras.
    void update(const std::vector<CrossCameraFrameInput> &batch,
                std::vector<CrossCameraTrackedObject> &tracked_objects);

    /// Also return one directly visualizable map position per global target.
    void update(const std::vector<CrossCameraFrameInput> &batch,
                std::vector<CrossCameraTrackedObject> &tracked_objects,
                std::vector<CrossCameraGlobalTarget> &global_targets);

  private:
    std::shared_ptr<CrossCameraTrackerPrivate> m_pHandle;
};

// ======================== 单应矩阵模块 ========================

/// 点对类型：first = 源平面点（如雷达坐标），second = 目标平面点（如图像坐标）
using PointPair = std::pair<cv::Point2f, cv::Point2f>;

class HomographyPrivate;
class Homography {
  public:
    Homography();
    ~Homography();

    Homography(const Homography &) = delete;
    Homography &operator=(const Homography &) = delete;

    /// 根据配对点计算单应矩阵（至少 4 对点），返回 3x3 cv::Mat
    cv::Mat compute(const std::vector<PointPair> &pairs);

    /// 同上，返回展平的 9 个 double（行优先）
    std::vector<double> compute_flat(const std::vector<PointPair> &pairs);

    /// 直接设置单应矩阵（9 个 double，行优先）
    void set_matrix(const std::vector<double> &values);
    /// 直接设置单应矩阵（3x3 cv::Mat）
    void set_matrix(const cv::Mat &mat);

    /// 批量投影：将源平面点通过单应矩阵映射到目标平面
    std::vector<cv::Point2f> project_points(const std::vector<cv::Point2f> &src);

    /// 单点投影
    cv::Point2f project_point(const cv::Point2f &pt);

    /// 获取当前单应矩阵
    cv::Mat get_matrix() const;

  private:
    std::shared_ptr<HomographyPrivate> m_pHandle;
};

std::string GetOptimalDevice();

} // namespace JHDeepCore
