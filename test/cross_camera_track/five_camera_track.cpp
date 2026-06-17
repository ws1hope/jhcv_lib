#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <memory>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace JHDeepCore;

/// 多路摄像头独立跟踪的诊断程序框架
/// 与 cross_camera_track/main.cpp 的区别：
///   1. 多路视频，每路一个独立 Tracker（track_id 各自计数）
///   2. 不做跨摄像头关联，不做单应矩阵映射
///   3. 画布只保留多路视频横向并排 + 各自跟踪框

static const int kCameraCount = 3;

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

/// 每路通道颜色（用于在画布顶部标识）
static cv::Scalar getColorForCamera(int camera_id)
{
    static const cv::Scalar cam_colors[] = {
        {0, 0, 255},   // 红
        {255, 0, 0},   // 蓝
        {0, 255, 0},   // 绿
        {0, 165, 255}, // 橙
        {255, 0, 255}  // 紫
    };
    return cam_colors[camera_id % 5];
}

/// 单路摄像头通道：含独立的 Tracker
struct CameraChannel {
    std::string name;
    cv::VideoCapture cap;
    int camera_id = 0;
    int width = 0;
    int height = 0;
    double fps = 0;
    int total_frames = 0;
    int x_offset = 0;           // 在合成画布中的横向偏移
    cv::Scalar border_color;

    std::unique_ptr<JHDeepCore::Tracker> tracker;  // 每路独立跟踪器

    CameraChannel(const std::string& video_path,
                  const std::string& channel_name,
                  int id)
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
        border_color = getColorForCamera(id);
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

/// 在画布的对应通道区域绘制该路跟踪结果
static void drawTrackedObjects(cv::Mat& canvas, const CameraChannel& ch,
                               const std::vector<JHDeepCore::TrackedObject>& tracked_objects)
{
    const int dx = ch.x_offset;
    for (const auto& obj : tracked_objects) {
        cv::Scalar color = getColorForTrackId(obj.track_id);

        cv::Rect bbox_shifted = obj.bbox;
        bbox_shifted.x += dx;
        cv::rectangle(canvas, bbox_shifted, color, 2);

        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::Point p1(obj.trajectory[j].x + dx, obj.trajectory[j].y);
                cv::Point p2(obj.trajectory[j + 1].x + dx, obj.trajectory[j + 1].y);
                cv::line(canvas, p1, p2, color, 2, cv::LINE_AA);
            }
            cv::Point last(obj.trajectory.back().x + dx, obj.trajectory.back().y);
            cv::circle(canvas, last, 4, color, -1, cv::LINE_AA);
        }

        std::string label = std::to_string(obj.track_id);
        int baseLine = 0;
        double labelScale = 1.6;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, labelScale, 4, &baseLine);
        int top = std::max(obj.bbox.y, textSize.height + 5);
        cv::rectangle(canvas,
                      cv::Point(bbox_shifted.x, top - textSize.height - 5),
                      cv::Point(bbox_shifted.x + textSize.width + 8, top + 4),
                      color, -1);
        cv::putText(canvas, label, cv::Point(bbox_shifted.x + 4, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, labelScale, cv::Scalar(255, 255, 255), 4, cv::LINE_AA);
    }

    // 通道边框（用于在画布上区分五路）
    cv::rectangle(canvas,
                  cv::Rect(dx, 0, ch.width - 1, ch.height - 1),
                  ch.border_color, 4);

    // 通道名称
    std::string cam_label = ch.name + " " + std::to_string(ch.width) + "x" + std::to_string(ch.height);
    cv::putText(canvas, cam_label, cv::Point(dx + 20, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.4, ch.border_color, 4, cv::LINE_AA);
}

int main(int argc, char* argv[])
{
    // ===== 默认参数 =====
    std::string model_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.onnx";
    std::string label_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.yaml";
    std::string output_path = "result/five_camera_track_result.avi";
    int device_id = -1;
    int skip_frames = 3;

    std::vector<std::string> video_paths(kCameraCount);
    for (int i = 0; i < kCameraCount; ++i) {
        video_paths[i] =
            "/Users/zhanghaining/2026code/jhcv_lib/images/test_video/camera" +
            std::to_string(i + 1) + ".mp4";
    }

    // ===== 命令行参数 =====
    // 用法:
    //   five_camera_track_test [model] [cam1] [cam2] [cam3] [label] [output] [device] [skip]
    if (argc >= 2) model_path = argv[1];
    for (int i = 0; i < kCameraCount; ++i) {
        if (argc >= 3 + i) video_paths[i] = argv[2 + i];
    }
    if (argc >= 2 + kCameraCount + 1) label_path = argv[2 + kCameraCount];
    if (argc >= 2 + kCameraCount + 2) output_path = argv[3 + kCameraCount];
    if (argc >= 2 + kCameraCount + 3) device_id = std::stoi(argv[4 + kCameraCount]);
    if (argc >= 2 + kCameraCount + 4) skip_frames = std::stoi(argv[5 + kCameraCount]);

    try {
        std::cout << "=== Five-Camera Independent Tracking ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Label: " << (label_path.empty() ? "Not specified" : label_path) << std::endl;
        for (int i = 0; i < kCameraCount; ++i) {
            std::cout << "Camera " << (i + 1) << ": " << video_paths[i] << std::endl;
        }
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Device ID: " << device_id << std::endl;
        std::cout << "Skip: process 1 frame every " << skip_frames << std::endl;
        std::cout << "================================" << std::endl;

        // 创建结果目录
        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        // ===== 共享检测器 =====
        std::cout << "Initializing detector..." << std::endl;
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready: " << detector.GetInputWidth() << "x"
                  << detector.GetInputHeight() << ", batch=" << detector.GetBatch() << std::endl;

        // ===== 五路通道 + 各自跟踪器 =====
        std::vector<std::unique_ptr<CameraChannel>> channels;
        channels.reserve(kCameraCount);
        float ref_fps = 0;
        for (int i = 0; i < kCameraCount; ++i) {
            auto ch = std::make_unique<CameraChannel>(
                video_paths[i], "Cam" + std::to_string(i + 1), i + 1);
            // 每路独立 Tracker，使用各自视频的实际 FPS（除以 skip_frames 还原真实处理节奏）
            float tracker_fps = static_cast<float>(ch->fps) /
                                static_cast<float>(std::max(skip_frames, 1));
            ch->tracker = std::make_unique<JHDeepCore::Tracker>(
                createTrackerConfig(), tracker_fps);
            std::cout << "Cam" << (i + 1) << ": " << ch->width << "x" << ch->height
                      << " @ " << ch->fps << " FPS, " << ch->total_frames << " frames"
                      << " (tracker_fps=" << tracker_fps << ")" << std::endl;
            if (i == 0) ref_fps = static_cast<float>(ch->fps);
            channels.push_back(std::move(ch));
        }

        // ===== 画布布局：五路横向并排 =====
        int top_height = 0;
        int top_width = 0;
        int total_frames = INT_MAX;
        for (auto& ch : channels) {
            top_height = std::max(top_height, ch->height);
            ch->x_offset = top_width;
            top_width += ch->width;
            total_frames = std::min(total_frames, ch->total_frames);
        }

        int canvas_width = top_width;
        int canvas_height = top_height;

        std::cout << "Canvas: " << canvas_width << "x" << canvas_height << std::endl;

        // 编码器选择：五路 2560 横向并排 = 12800 宽，超出 MPEG-4/H.264 上限，
        // 改用 MJPG（Motion JPEG），它几乎没有尺寸限制（>16K），代价是文件体积较大。
        // 统一输出为 .avi + MJPG。
        std::string ext = output_path.substr(output_path.find_last_of('.'));
        if (ext != ".avi") {
            output_path = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
        }
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

        cv::VideoWriter writer(output_path, fourcc, ref_fps,
                               cv::Size(canvas_width, canvas_height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video: " << output_path << std::endl;
            std::cerr << "Hint: canvas size " << canvas_width << "x" << canvas_height
                      << " is too large even for MJPG. Check ffmpeg build or reduce sources."
                      << std::endl;
            return 1;
        }

        std::cout << "Starting..." << std::endl;

        // ===== 主循环 =====
        std::vector<cv::Mat> frames(kCameraCount);
        std::vector<std::vector<JHDeepCore::TrackedObject>> tracked_per_cam(kCameraCount);

        int total_count = 0;
        int processed_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            // 读五路帧
            bool any_valid = false;
            for (int i = 0; i < kCameraCount; ++i) {
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
            for (int i = 0; i < kCameraCount; ++i) {
                frames[i].copyTo(canvas(cv::Rect(channels[i]->x_offset, 0,
                                                  channels[i]->width, channels[i]->height)));
            }

            // 各路独立检测 + 跟踪（单张推理，模型可能只支持 batch=1）
            auto t_det_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < kCameraCount; ++i) {
                std::vector<cv::Mat> detector_input = {frames[i]};
                std::vector<JHDeepCore::DetectionResult> det_results;
                detector.process(detector_input, det_results);
                if (det_results.size() != 1) {
                    throw std::runtime_error(
                        "Detector did not return one result for camera " +
                        std::to_string(channels[i]->camera_id));
                }
                std::vector<JHDeepCore::Detection> dets = convertDetections(det_results[0]);
                channels[i]->tracker->update(dets, frames[i], tracked_per_cam[i]);
            }
            auto t_det_end = std::chrono::high_resolution_clock::now();

            // 绘制各路结果
            for (int i = 0; i < kCameraCount; ++i) {
                drawTrackedObjects(canvas, *channels[i], tracked_per_cam[i]);
            }

            // 全局帧号
            std::string frame_label = "Frame: " + std::to_string(total_count) +
                                      "  Processed: " + std::to_string(processed_count);
            cv::putText(canvas, frame_label, cv::Point(30, 120),
                        cv::FONT_HERSHEY_SIMPLEX, 2.4, cv::Scalar(0, 0, 255), 6, cv::LINE_AA);

            writer.write(canvas);
            processed_count++;

            auto t_frame_end = std::chrono::high_resolution_clock::now();
            auto det_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_det_end - t_det_start).count();
            auto frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_frame_end - t_frame_start).count();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_frame_end - start_time).count();
            float progress = static_cast<float>(total_count) / total_frames * 100.0f;

            // 各路跟踪数统计
            std::ostringstream tracked_summary;
            for (int i = 0; i < kCameraCount; ++i) {
                if (i) tracked_summary << " ";
                tracked_summary << "C" << (i + 1) << "=" << tracked_per_cam[i].size();
            }

            std::cout << "Frame " << total_count << "/" << total_frames
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | det+track:" << det_ms << "ms total:" << frame_ms << "ms"
                      << " | " << tracked_summary.str()
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }

        // 释放
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
