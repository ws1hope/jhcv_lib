#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>
#include "../../src/third_party/json.hpp"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace JHDeepCore;
using json = nlohmann::json;

/// 五路摄像头跨摄像头跟踪（仿 cross_camera_track/main.cpp）
/// 与之前版本的区别：
///   1. 从 biaoding_calibration.json 加载 5 路标定（camera_id: 1,2,3,4,6）
///   2. 使用 CrossCameraTracker 做跨摄像头关联（共享 target_id）
///   3. 画布三层：顶部 5 路视频并排 + 中部全局 EMA 点 + 底部每路映射点

static const int kExpectedCameras = 5;

static bool ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

static cv::Scalar getColorForTrackId(size_t track_id)
{
    static const cv::Scalar colors[] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 0, 255}, {255, 128, 0},
        {128, 255, 0}, {0, 128, 255}, {255, 0, 128}, {0, 255, 128}
    };
    return colors[track_id % 12];
}

static cv::Scalar getColorForCamera(int camera_idx)
{
    static const cv::Scalar cam_colors[] = {
        {0, 0, 255},   // 红
        {255, 0, 0},   // 蓝
        {0, 255, 0},   // 绿
        {0, 165, 255}, // 橙
        {255, 0, 255}  // 紫
    };
    return cam_colors[camera_idx % 5];
}

/// 单路视频通道：仅承担读取与绘制，跟踪交给 CrossCameraTracker
struct CameraChannel {
    std::string name;
    cv::VideoCapture cap;
    CameraId camera_id = 0;
    int width = 0;
    int height = 0;
    double fps = 0;
    int total_frames = 0;
    int x_offset = 0;
    cv::Scalar border_color;
    cv::Scalar map_color;

    CameraChannel(const std::string& video_path,
                  const std::string& channel_name,
                  CameraId id)
        : name(channel_name), camera_id(id)
    {
        cap.open(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open video: " + video_path);
        }
        fps = cap.get(cv::CAP_PROP_FPS);
        width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    }
};

static JHDeepCore::TrackerConfig createTrackerConfig()
{
    JHDeepCore::TrackerConfig cfg;
    cfg.tracker_type = JHDeepCore::TrackerType::ByteTrack;
    cfg.distance_type = JHDeepCore::TrackDistanceType::IoU;
    cfg.distance_threshold = 0.6f;
    cfg.bytetrack_track_thresh = 0.5f;
    cfg.bytetrack_high_thresh = 0.5f;
    cfg.bytetrack_match_thresh = 0.8f;
    return cfg;
}

static std::vector<JHDeepCore::Detection> convertDetections(
    const JHDeepCore::DetectionResult& det_result)
{
    std::vector<JHDeepCore::Detection> detections;
    for (const auto& det : det_result.detections) {
        JHDeepCore::Detection d;
        d.bbox = det.bbox;
        d.confidence = det.confidence;
        d.class_id = det.class_id;
        d.class_name = det.class_name;
        detections.push_back(d);
    }
    return detections;
}

/// 标定数据：从 biaoding_calibration.json 读取
struct CalibrationData {
    int map_width = 0;
    int map_height = 0;
    /// 每路摄像头的 camera_id 与点对（按 JSON 中的顺序保留）
    std::vector<CameraId> camera_ids;
    std::vector<std::vector<CalibrationPointPair>> points_per_cam;
};

static bool loadCalibration(const std::string& json_path, CalibrationData& out)
{
    std::ifstream in(json_path);
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open calibration JSON: " << json_path << std::endl;
        return false;
    }
    json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to parse calibration JSON: " << e.what() << std::endl;
        return false;
    }

    out.map_width = j.value("map_width", 0);
    out.map_height = j.value("map_height", 0);
    if (out.map_width <= 0 || out.map_height <= 0) {
        std::cerr << "Error: Invalid map size in calibration: "
                  << out.map_width << "x" << out.map_height << std::endl;
        return false;
    }

    for (const auto& cam : j["cameras"]) {
        CameraId cid = cam.value("camera_id", static_cast<CameraId>(0));
        std::vector<CalibrationPointPair> pairs;
        for (const auto& p : cam["points"]) {
            CalibrationPointPair pair;
            pair.image_point = cv::Point2f(
                p["image_point"][0].get<float>(),
                p["image_point"][1].get<float>());
            pair.map_point = cv::Point2f(
                p["map_point"][0].get<float>(),
                p["map_point"][1].get<float>());
            pairs.push_back(pair);
        }
        if (pairs.size() < 4) {
            std::cerr << "Error: Camera " << cid << " has only " << pairs.size()
                      << " pairs (need >= 4 for homography)" << std::endl;
            return false;
        }
        out.camera_ids.push_back(cid);
        out.points_per_cam.push_back(std::move(pairs));
    }
    return true;
}

/// 绘制单路局部跟踪（在顶部视频区域内）
static void drawTrackedObjects(
    cv::Mat& canvas, CameraChannel& ch,
    const std::vector<JHDeepCore::CrossCameraTrackedObject>& tracked_objects)
{
    for (const auto& cross_obj : tracked_objects) {
        if (cross_obj.camera_id != ch.camera_id) continue;
        const auto& obj = cross_obj.local_track;
        cv::Scalar color = getColorForTrackId(cross_obj.target_id);

        cv::Rect bbox_shifted = obj.bbox;
        bbox_shifted.x += ch.x_offset;
        cv::rectangle(canvas, bbox_shifted, color, 2);

        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::Point p1(obj.trajectory[j].x + ch.x_offset, obj.trajectory[j].y);
                cv::Point p2(obj.trajectory[j + 1].x + ch.x_offset, obj.trajectory[j + 1].y);
                cv::line(canvas, p1, p2, color, 2, cv::LINE_AA);
            }
            cv::Point last(obj.trajectory.back().x + ch.x_offset, obj.trajectory.back().y);
            cv::circle(canvas, last, 4, color, -1, cv::LINE_AA);
        }

        std::string label = std::to_string(cross_obj.target_id);
        int baseLine = 0;
        double labelScale = 2.0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, labelScale, 6, &baseLine);
        int top = std::max(obj.bbox.y, textSize.height + 5);
        cv::rectangle(canvas,
                      cv::Point(bbox_shifted.x, top - textSize.height - 5),
                      cv::Point(bbox_shifted.x + textSize.width + 8, top + 4),
                      color, -1);
        cv::putText(canvas, label, cv::Point(bbox_shifted.x + 4, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, labelScale, cv::Scalar(255, 255, 255), 6, cv::LINE_AA);

        cv::putText(canvas, "C" + std::to_string(ch.camera_id) + "L" + std::to_string(obj.track_id),
                    cv::Point(bbox_shifted.x, top + 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv::LINE_AA);
    }

    // 通道边框
    cv::rectangle(canvas,
                  cv::Rect(ch.x_offset, 0, ch.width - 1, ch.height - 1),
                  ch.border_color, 4);
    std::string cam_label = ch.name + " " + std::to_string(ch.width) + "x" + std::to_string(ch.height);
    cv::putText(canvas, cam_label, cv::Point(ch.x_offset + 20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, ch.border_color, 4, cv::LINE_AA);
}

/// 绘制全局 EMA 点（中部白底区域）
static void drawGlobalTargets(
    cv::Mat& canvas,
    const std::vector<JHDeepCore::CrossCameraGlobalTarget>& global_targets,
    int top_height)
{
    for (const auto& target : global_targets) {
        const cv::Point canvas_point(
            cvRound(target.smoothed_mapped_point.x),
            cvRound(target.smoothed_mapped_point.y) + top_height);
        const cv::Scalar color = getColorForTrackId(target.target_id);
        cv::circle(canvas, canvas_point, 10, color, -1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(target.target_id),
                    cv::Point(canvas_point.x - 18, canvas_point.y - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 1.4, color, 4, cv::LINE_AA);
    }
}

int main(int argc, char* argv[])
{
    // ===== 默认参数 =====
    std::string model_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.onnx";
    std::string label_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.yaml";
    std::string calibration_path =
        "/Users/zhanghaining/2026code/jhcv_lib/result/biaoding_calibration.json";
    std::string video_dir =
        "/Users/zhanghaining/2026code/jhcv_lib/images/test_video";
    std::string output_path = "result/five_camera_cross_result.avi";
    int device_id = -1;
    int skip_frames = 3;
    float link_distance = 200.0f;

    // ===== 命令行参数 =====
    // 用法:
    //   five_camera_track_test [model] [label] [calibration] [video_dir] [output] [device] [skip] [link_dist]
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) label_path = argv[2];
    if (argc >= 4) calibration_path = argv[3];
    if (argc >= 5) video_dir = argv[4];
    if (argc >= 6) output_path = argv[5];
    if (argc >= 7) device_id = std::stoi(argv[6]);
    if (argc >= 8) skip_frames = std::stoi(argv[7]);
    if (argc >= 9) link_distance = std::stof(argv[8]);

    try {
        std::cout << "=== Five-Camera Cross-Camera Tracking ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Label: " << label_path << std::endl;
        std::cout << "Calibration: " << calibration_path << std::endl;
        std::cout << "Video dir: " << video_dir << std::endl;
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Device ID: " << device_id << std::endl;
        std::cout << "Skip: process 1 frame every " << skip_frames << std::endl;
        std::cout << "Link distance: " << link_distance << std::endl;
        std::cout << "================================" << std::endl;

        // 创建结果目录
        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        // ===== 1. 加载标定 =====
        CalibrationData calib;
        if (!loadCalibration(calibration_path, calib)) {
            return 1;
        }
        if (static_cast<int>(calib.camera_ids.size()) != kExpectedCameras) {
            std::cerr << "Warning: expected " << kExpectedCameras
                      << " cameras, got " << calib.camera_ids.size() << std::endl;
        }
        std::cout << "Map size: " << calib.map_width << "x" << calib.map_height << std::endl;
        for (size_t i = 0; i < calib.camera_ids.size(); ++i) {
            std::cout << "  Camera id=" << calib.camera_ids[i]
                      << " with " << calib.points_per_cam[i].size() << " calibration points"
                      << std::endl;
        }

        // ===== 2. 共享检测器 =====
        std::cout << "Initializing detector..." << std::endl;
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready: " << detector.GetInputWidth() << "x"
                  << detector.GetInputHeight() << ", batch=" << detector.GetBatch() << std::endl;

        // ===== 3. 创建五路通道 =====
        std::vector<std::unique_ptr<CameraChannel>> channels;
        channels.reserve(calib.camera_ids.size());
        float ref_fps = 0;
        for (size_t i = 0; i < calib.camera_ids.size(); ++i) {
            CameraId cid = calib.camera_ids[i];
            std::string video_path = video_dir + "/camera" + std::to_string(cid) + ".mp4";
            auto ch = std::make_unique<CameraChannel>(
                video_path, "Cam" + std::to_string(cid), cid);
            ch->border_color = getColorForCamera(static_cast<int>(i));
            ch->map_color = ch->border_color;
            std::cout << "Cam" << cid << ": " << ch->width << "x" << ch->height
                      << " @ " << ch->fps << " FPS, " << ch->total_frames << " frames" << std::endl;
            if (i == 0) ref_fps = static_cast<float>(ch->fps);
            channels.push_back(std::move(ch));
        }

        // ===== 4. 配置 CrossCameraTracker =====
        CrossCameraTrackerConfig cross_config;
        cross_config.tracker_config = createTrackerConfig();
        cross_config.enable_log = true;
        cross_config.log_directory = "logs";
        for (size_t i = 0; i < channels.size(); ++i) {
            CrossCameraChannelConfig ch_cfg;
            ch_cfg.camera_id = channels[i]->camera_id;
            ch_cfg.tracker_fps = static_cast<float>(channels[i]->fps) /
                                 static_cast<float>(std::max(skip_frames, 1));
            ch_cfg.calibration_points = calib.points_per_cam[i];
            cross_config.channels.push_back(std::move(ch_cfg));
        }
        // 邻接关系：按通道顺序链式相连
        for (size_t i = 0; i + 1 < channels.size(); ++i) {
            cross_config.links.push_back({
                channels[i]->camera_id,
                channels[i + 1]->camera_id,
                link_distance
            });
        }
        CrossCameraTracker cross_tracker(cross_config);

        // ===== 5. 画布布局 =====
        int top_height = 0;
        int top_width = 0;
        int total_frames = INT_MAX;
        for (auto& ch : channels) {
            top_height = std::max(top_height, ch->height);
            ch->x_offset = top_width;
            top_width += ch->width;
            total_frames = std::min(total_frames, ch->total_frames);
        }
        int map_width = calib.map_width;
        int map_height = calib.map_height;
        int canvas_width = std::max(top_width, map_width);
        int canvas_height = top_height + map_height * 2;
        std::cout << "Canvas: " << canvas_width << "x" << canvas_height << std::endl;

        // MJPG 编码（画布尺寸超过 MPEG-4 / H.264 上限，必须用 MJPG）
        std::string ext = output_path.substr(output_path.find_last_of('.'));
        if (ext != ".avi") {
            output_path = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
        }
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        cv::VideoWriter writer(output_path, fourcc, ref_fps,
                               cv::Size(canvas_width, canvas_height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video: " << output_path << std::endl;
            return 1;
        }

        std::cout << "Starting..." << std::endl;

        // ===== 主循环 =====
        std::vector<cv::Mat> frames(channels.size());
        int total_count = 0;
        int processed_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            bool any_valid = false;
            for (size_t i = 0; i < channels.size(); ++i) {
                if (channels[i]->cap.read(frames[i]) && !frames[i].empty()) {
                    any_valid = true;
                } else {
                    frames[i] = cv::Mat::zeros(channels[i]->height, channels[i]->width, CV_8UC3);
                }
            }
            if (!any_valid) break;
            total_count++;
            if (total_count % skip_frames != 0) continue;

            auto t_frame_start = std::chrono::high_resolution_clock::now();

            // 画布：黑色背景
            cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(0, 0, 0));
            // 顶部：五路视频
            for (size_t i = 0; i < channels.size(); ++i) {
                frames[i].copyTo(canvas(cv::Rect(channels[i]->x_offset, 0,
                                                  channels[i]->width, channels[i]->height)));
            }
            // 中部：全局 EMA 点白底
            cv::Mat global_roi = canvas(cv::Rect(0, top_height, map_width, map_height));
            global_roi.setTo(cv::Scalar(255, 255, 255));
            cv::putText(global_roi, "Global EMA Targets", cv::Point(20, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            // 底部：每路映射点白底
            cv::Mat cam_roi = canvas(cv::Rect(0, top_height + map_height, map_width, map_height));
            cam_roi.setTo(cv::Scalar(255, 255, 255));
            cv::putText(cam_roi, "Per-Camera Mapped Points", cv::Point(20, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);

            // 逐路检测（模型通常 batch=1）
            std::vector<CrossCameraFrameInput> cross_inputs;
            cross_inputs.reserve(channels.size());

            auto t_det_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < channels.size(); ++i) {
                std::vector<cv::Mat> detector_input = {frames[i]};
                std::vector<JHDeepCore::DetectionResult> det_results;
                detector.process(detector_input, det_results);
                if (det_results.size() != 1) {
                    throw std::runtime_error(
                        "Detector did not return one result for camera " +
                        std::to_string(channels[i]->camera_id));
                }
                CrossCameraFrameInput input;
                input.camera_id = channels[i]->camera_id;
                input.frame = frames[i];
                input.detections = convertDetections(det_results[0]);
                cross_inputs.push_back(std::move(input));
            }
            auto t_det_end = std::chrono::high_resolution_clock::now();

            // 跨摄像头关联 + 全局目标
            auto t_track_start = std::chrono::high_resolution_clock::now();
            std::vector<CrossCameraTrackedObject> tracked_objects;
            std::vector<CrossCameraGlobalTarget> global_targets;
            cross_tracker.update(cross_inputs, tracked_objects, global_targets);
            auto t_track_end = std::chrono::high_resolution_clock::now();

            // 绘制
            for (auto& ch : channels) {
                drawTrackedObjects(canvas, *ch, tracked_objects);
            }
            drawGlobalTargets(canvas, global_targets, top_height);

            // 底部每路 mapped_point
            int cam_map_y_offset = top_height + map_height;
            for (const auto& cross_obj : tracked_objects) {
                for (const auto& ch : channels) {
                    if (cross_obj.camera_id != ch->camera_id) continue;
                    cv::Point pt(cvRound(cross_obj.mapped_point.x),
                                 cvRound(cross_obj.mapped_point.y) + cam_map_y_offset);
                    cv::circle(canvas, pt, 6, ch->map_color, -1, cv::LINE_AA);
                    std::string label = "C" + std::to_string(cross_obj.camera_id) +
                                        "L" + std::to_string(cross_obj.local_track.track_id) +
                                        " G" + std::to_string(cross_obj.target_id);
                    cv::putText(canvas, label, cv::Point(pt.x + 10, pt.y + 8),
                                cv::FONT_HERSHEY_SIMPLEX, 0.9, ch->map_color, 2, cv::LINE_AA);
                }
            }

            std::string frame_label = "Frame: " + std::to_string(total_count) +
                                      "  Processed: " + std::to_string(processed_count);
            cv::putText(canvas, frame_label, cv::Point(30, 120),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 6, cv::LINE_AA);

            writer.write(canvas);
            processed_count++;

            auto t_frame_end = std::chrono::high_resolution_clock::now();
            auto det_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_det_end - t_det_start).count();
            auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_track_end - t_track_start).count();
            auto frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_frame_end - t_frame_start).count();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_frame_end - start_time).count();
            float progress = static_cast<float>(total_count) / total_frames * 100.0f;

            std::cout << "Frame " << total_count << "/" << total_frames
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | det:" << det_ms << "ms track:" << track_ms
                      << "ms total:" << frame_ms << "ms"
                      << " | tracked=" << tracked_objects.size()
                      << " globals=" << global_targets.size()
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }

        for (auto& ch : channels) ch->cap.release();
        writer.release();

        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        double avg_fps = (total_sec > 0) ? static_cast<double>(processed_count) / total_sec : 0;

        std::cout << "================================" << std::endl;
        std::cout << "Completed! " << processed_count << " frames in " << total_sec << "s"
                  << " (" << std::fixed << std::setprecision(2) << avg_fps << " FPS)" << std::endl;
        std::cout << "Output: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
