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

/// 单个摄像头的检测+跟踪+单应矩阵映射
struct CameraChannel {
    std::string name;
    cv::VideoCapture cap;
    JHDeepCore::Tracker tracker;
    Homography homo;
    cv::Scalar map_color;
    int width = 0;
    int height = 0;
    int x_offset = 0;
    double fps = 0;
    int total_frames = 0;

    CameraChannel(const std::string& video_path,
                  const std::string& channel_name,
                  float tracker_fps)
        : name(channel_name), tracker(createTrackerConfig(), tracker_fps)
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

private:
    static JHDeepCore::TrackerConfig createTrackerConfig() {
        JHDeepCore::TrackerConfig cfg;
        cfg.tracker_type = JHDeepCore::TrackerType::ByteTrack;
        cfg.distance_type = JHDeepCore::TrackDistanceType::IoU;
        cfg.distance_threshold = 0.6f;
        cfg.bytetrack_track_thresh = 0.5f;
        cfg.bytetrack_high_thresh = 0.5f;
        cfg.bytetrack_match_thresh = 0.8f;
        return cfg;
    }
};

static std::vector<JHDeepCore::Detection> convertDetections(const JHDeepCore::DetectionResult& det_result)
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

/// 在上方视频区域绘制跟踪结果，在下方白底区域绘制映射点
static void drawTrackedObjects(cv::Mat& canvas, CameraChannel& ch,
                               const std::vector<JHDeepCore::TrackedObject>& tracked_objects,
                               int top_height)
{
    for (const auto& obj : tracked_objects) {
        cv::Scalar color = getColorForTrackId(obj.track_id);

        // 上方视频区域：跟踪框（带偏移）
        cv::Rect bbox_shifted = obj.bbox;
        bbox_shifted.x += ch.x_offset;
        cv::rectangle(canvas, bbox_shifted, color, 2);

        // 轨迹线
        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::Point p1(obj.trajectory[j].x + ch.x_offset, obj.trajectory[j].y);
                cv::Point p2(obj.trajectory[j + 1].x + ch.x_offset, obj.trajectory[j + 1].y);
                cv::line(canvas, p1, p2, color, 2, cv::LINE_AA);
            }
            cv::Point last(obj.trajectory.back().x + ch.x_offset, obj.trajectory.back().y);
            cv::circle(canvas, last, 4, color, -1, cv::LINE_AA);
        }

        // 标签: 只显示 ID
        std::string label = std::to_string(obj.track_id);
        int baseLine = 0;
        double labelScale = 2.4;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, labelScale, 8, &baseLine);
        int top = std::max(obj.bbox.y, textSize.height + 5);
        cv::rectangle(canvas,
                      cv::Point(bbox_shifted.x, top - textSize.height - 5),
                      cv::Point(bbox_shifted.x + textSize.width + 8, top + 4),
                      color, -1);
        cv::putText(canvas, label, cv::Point(bbox_shifted.x + 4, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, labelScale, cv::Scalar(255, 255, 255), 8, cv::LINE_AA);

        // 中心点映射到白底区域
        cv::Point2f center(obj.bbox.x + obj.bbox.width / 2.0f,
                           obj.bbox.y + obj.bbox.height / 2.0f);
        cv::Point2f mapped = ch.homo.project_point(center);
        cv::Point canvas_mapped(static_cast<int>(mapped.x),
                                static_cast<int>(mapped.y) + top_height);
        cv::circle(canvas, canvas_mapped, 8, ch.map_color, -1, cv::LINE_AA);

        // 白底区域：映射点上方绘制 ID
        cv::putText(canvas, std::to_string(obj.track_id),
                    cv::Point(canvas_mapped.x - 15, canvas_mapped.y - 16),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, ch.map_color, 4, cv::LINE_AA);
    }
}

int main(int argc, char* argv[])
{
    std::string model_path = "";
    std::string label_path = "";
    std::string output_path = "result/cross_camera_result.avi";
    int device_id = 0;
    int skip_frames = 3;

    std::vector<std::string> video_paths = {
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td1.mp4",
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td2.mp4",
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td3.mp4"
    };

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) video_paths[0] = argv[2];
    if (argc >= 4) video_paths[1] = argv[3];
    if (argc >= 5) video_paths[2] = argv[4];
    if (argc >= 6) label_path = argv[5];
    if (argc >= 7) output_path = argv[6];
    if (argc >= 8) device_id = std::stoi(argv[7]);
    if (argc >= 9) skip_frames = std::stoi(argv[8]);

    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> [video1] [video2] [video3] [label_path] [output_path] [device_id] [skip_frames]"
                  << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Cross-Camera Tracking with Homography ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        for (size_t i = 0; i < video_paths.size(); ++i) {
            std::cout << "Camera " << (i + 1) << ": " << video_paths[i] << std::endl;
        }
        std::cout << "Output: " << output_path << std::endl;

        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        // 共享检测器
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready: " << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

        // 单应矩阵点对
        std::vector<PointPair> pairs1 = {
            {{1692, 153},  {750, 836}},
            {{1892, 231},  {750, 702}},
            {{818,  1439}, {2505, 702}},
            {{292,  1175}, {2505, 836}},
        };
        std::vector<PointPair> pairs2 = {
            {{2430, 1091}, {2505, 836}},
            {{2008, 1439}, {2505, 702}},
            {{882,  0},    {4800, 702}},
            {{1192, 0},    {4800, 836}},
        };
        std::vector<PointPair> pairs3 = {
            {{115, 337},   {4575, 836}},
            {{263, 307},   {4575, 702}},
            {{2399, 380},  {7084, 234}},
            {{577, 1439},  {7084, 839}},
        };
        std::vector<std::vector<PointPair>> all_pairs = {pairs1, pairs2, pairs3};
        std::vector<cv::Scalar> map_colors = {
            cv::Scalar(0, 0, 255),     // 红色
            cv::Scalar(255, 0, 0),     // 蓝色
            cv::Scalar(0, 255, 0),     // 绿色
        };

        // 初始化三路通道
        std::vector<std::unique_ptr<CameraChannel>> channels;
        float ref_fps = 0;
        for (size_t i = 0; i < video_paths.size(); ++i) {
            auto ch = std::make_unique<CameraChannel>(
                video_paths[i], "Cam" + std::to_string(i + 1), 25.0f);
            ch->map_color = map_colors[i];

            cv::Mat H = ch->homo.compute(all_pairs[i]);
            if (H.empty()) {
                std::cerr << "Error: Failed to compute homography for Cam" << (i + 1) << std::endl;
                return 1;
            }
            std::cout << "Cam" << (i + 1) << ": " << ch->width << "x" << ch->height
                      << " @ " << ch->fps << " FPS, " << ch->total_frames << " frames"
                      << " | Homography OK" << std::endl;
            if (i == 0) ref_fps = static_cast<float>(ch->fps);
            channels.push_back(std::move(ch));
        }

        // 计算布局：上方三视频并排，下方白底7500x1000
        int top_height = 0;
        int top_width = 0;
        int total_frames = INT_MAX;
        for (auto& ch : channels) {
            top_height = std::max(top_height, ch->height);
            ch->x_offset = top_width;
            top_width += ch->width;
            total_frames = std::min(total_frames, ch->total_frames);
        }

        int map_width = 7500;
        int map_height = 1000;
        int canvas_width = top_width;
        int canvas_height = top_height + map_height;

        std::string ext = output_path.substr(output_path.find_last_of('.'));
        if (ext != ".avi") {
            output_path = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
        }
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                               ref_fps, cv::Size(canvas_width, canvas_height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video" << std::endl;
            return 1;
        }

        std::cout << "Canvas: " << canvas_width << "x" << canvas_height << std::endl;
        std::cout << "Starting..." << std::endl;

        // 主循环
        std::vector<cv::Mat> frames(channels.size());
        int total_count = 0;
        int processed_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            // 读取三路帧
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

            // 创建画布
            cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(0, 0, 0));

            // 上方放置三路视频（原始尺寸）
            for (size_t i = 0; i < channels.size(); ++i) {
                frames[i].copyTo(canvas(cv::Rect(channels[i]->x_offset, 0,
                                                  channels[i]->width, channels[i]->height)));
            }

            // 左下白底
            cv::Mat white_roi = canvas(cv::Rect(0, top_height, map_width, map_height));
            white_roi.setTo(cv::Scalar(255, 255, 255));

            // 逐路检测+跟踪+绘制
            for (size_t i = 0; i < channels.size(); ++i) {
                // 检测
                std::vector<cv::Mat> batch = {frames[i]};
                std::vector<JHDeepCore::DetectionResult> det_results;
                detector.process(batch, det_results);

                std::vector<JHDeepCore::Detection> detections;
                if (!det_results.empty()) {
                    detections = convertDetections(det_results[0]);
                }

                // 跟踪
                std::vector<JHDeepCore::TrackedObject> tracked;
                channels[i]->tracker.update(detections, frames[i], tracked);

                // 绘制跟踪结果 + 映射点
                drawTrackedObjects(canvas, *channels[i], tracked, top_height);
            }

            writer.write(canvas);
            processed_count++;

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            float progress = static_cast<float>(total_count) / total_frames * 100.0f;

            std::cout << "Frame " << total_count << "/" << total_frames
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | Processed: " << processed_count
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
