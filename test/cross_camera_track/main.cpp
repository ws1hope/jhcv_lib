#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

/// 确保目录存在
static bool ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

/// 根据跟踪ID生成固定颜色
static cv::Scalar getColorForTrackId(size_t track_id)
{
    static const cv::Scalar colors[] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 0, 255}, {255, 128, 0},
        {128, 255, 0}, {0, 128, 255}, {255, 0, 128}, {0, 255, 128}
    };
    return colors[track_id % 12];
}

/// 单个摄像头的检测+跟踪+标注
struct CameraChannel {
    std::string name;
    cv::VideoCapture cap;
    JHDeepCore::Tracker tracker;
    int width = 0;
    int height = 0;
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

/// 将检测结果转换为跟踪所需的Detection格式
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

/// 在单路画面上绘制跟踪结果和帧号
static void drawChannelResult(cv::Mat& frame,
                               const std::vector<JHDeepCore::TrackedObject>& tracked_objects,
                               const std::string& channel_name,
                               int frame_num)
{
    // 顶部通道名称
    int baseLine = 0;
    double nameScale = (frame.cols < 1000) ? 0.7 : 0.9;
    cv::Size nameSize = cv::getTextSize(channel_name, cv::FONT_HERSHEY_SIMPLEX, nameScale, 2, &baseLine);
    cv::Rect nameBg(10, 30 - nameSize.height - 4, nameSize.width + 16, nameSize.height + 12);
    cv::Mat nameOverlay = frame.clone();
    cv::rectangle(nameOverlay, nameBg, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(nameOverlay, 0.7, frame, 0.3, 0, frame);
    cv::putText(frame, channel_name, cv::Point(18, 30 - 2),
                cv::FONT_HERSHEY_SIMPLEX, nameScale, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    // 帧号
    if (frame_num > 0) {
        std::string frame_text = "Frame: " + std::to_string(frame_num);
        double fontScale = (frame.cols < 1000) ? 1.0 : 1.2;
        int y_pos = 70;
        cv::Size textSize = cv::getTextSize(frame_text, cv::FONT_HERSHEY_SIMPLEX, fontScale, 3, &baseLine);
        cv::Rect bg_rect(10, y_pos - textSize.height - 6, textSize.width + 16, textSize.height + 14);
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, bg_rect, cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
        cv::putText(frame, frame_text, cv::Point(18, y_pos),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    // 跟踪框 + 轨迹 + 标签
    for (const auto& obj : tracked_objects) {
        cv::Scalar color = getColorForTrackId(obj.track_id);
        cv::rectangle(frame, obj.bbox, color, 2);

        // 轨迹
        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::line(frame, obj.trajectory[j], obj.trajectory[j + 1], color, 2, cv::LINE_AA);
            }
            cv::circle(frame, obj.trajectory.back(), 4, color, -1, cv::LINE_AA);
        }

        // 标签: ID_置信度
        std::ostringstream oss;
        oss << obj.track_id << "_" << std::fixed << std::setprecision(2) << obj.confidence;
        std::string label = oss.str();

        double labelScale = (frame.cols < 1000) ? 0.6 : 0.8;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, labelScale, 2, &baseLine);
        int top = std::max(obj.bbox.y, textSize.height + 5);
        cv::rectangle(frame,
                      cv::Point(obj.bbox.x, top - textSize.height - 3),
                      cv::Point(obj.bbox.x + textSize.width + 4, top + 2),
                      color, -1);
        cv::putText(frame, label, cv::Point(obj.bbox.x + 2, top - 1),
                    cv::FONT_HERSHEY_SIMPLEX, labelScale,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}

int main(int argc, char* argv[])
{
    // 默认参数
    std::string model_path = "";
    std::string label_path = "";
    std::string output_path = "result/cross_camera_result.avi";
    int device_id = 0;
    int skip_frames = 3;
    int target_height = 720;  // 统一缩放高度，保证拼接对齐

    std::vector<std::string> video_paths = {
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td1.mp4",
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td2.mp4",
        "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td3.mp4"
    };

    // 解析命令行参数
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
        std::cerr << "  Cross-camera detection & tracking, horizontal concatenation output" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Cross-Camera Detection & Tracking ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        for (size_t i = 0; i < video_paths.size(); ++i) {
            std::cout << "Camera " << (i + 1) << ": " << video_paths[i] << std::endl;
        }
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Skip: every " << skip_frames << " frames" << std::endl;
        std::cout << "================================" << std::endl;

        // 创建结果目录
        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        // 初始化共享检测器
        std::cout << "Initializing detector..." << std::endl;
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready. Input: "
                  << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

        // 初始化三路通道（各自独立跟踪器）
        std::vector<std::unique_ptr<CameraChannel>> channels;
        float ref_fps = 0;
        for (size_t i = 0; i < video_paths.size(); ++i) {
            auto ch = std::make_unique<CameraChannel>(
                video_paths[i], "Cam" + std::to_string(i + 1), 25.0f);
            std::cout << "Cam" << (i + 1) << ": "
                      << ch->width << "x" << ch->height << " @ " << ch->fps << " FPS, "
                      << ch->total_frames << " frames" << std::endl;
            if (i == 0) ref_fps = static_cast<float>(ch->fps);
            channels.push_back(std::move(ch));
        }

        // 计算拼接后的分辨率
        // 所有路统一缩放到 target_height，保持宽高比
        int total_width = 0;
        std::vector<double> scale_ratios;
        for (auto& ch : channels) {
            double ratio = static_cast<double>(target_height) / ch->height;
            scale_ratios.push_back(ratio);
            total_width += static_cast<int>(ch->width * ratio);
        }
        cv::Size output_size(total_width, target_height);

        std::cout << "Concat output: " << total_width << "x" << target_height << std::endl;

        // 创建视频写入器
        std::string ext = output_path.substr(output_path.find_last_of('.'));
        int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        if (ext != ".avi") {
            output_path = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
        }
        cv::VideoWriter writer(output_path, fourcc, ref_fps, output_size);
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video" << std::endl;
            return 1;
        }

        // 主循环
        int total_frame_count = 0;
        int processed_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        // 找到最大帧数作为循环终止条件
        int max_frames = 0;
        for (auto& ch : channels) {
            max_frames = std::max(max_frames, ch->total_frames);
        }

        std::cout << "Starting cross-camera processing..." << std::endl;

        while (true) {
            total_frame_count++;

            // 跳帧
            if (total_frame_count % skip_frames != 0) {
                // 仍然需要读取（消耗）帧，但不处理
                bool any_has_frame = false;
                for (auto& ch : channels) {
                    cv::Mat dummy;
                    if (ch->cap.read(dummy)) any_has_frame = true;
                }
                if (!any_has_frame) break;
                continue;
            }

            auto frame_start = std::chrono::high_resolution_clock::now();

            std::vector<cv::Mat> channel_frames;
            std::vector<cv::Mat> scaled_frames;

            // 读取三路帧
            bool any_valid = false;
            for (auto& ch : channels) {
                cv::Mat frame;
                if (ch->cap.read(frame) && !frame.empty()) {
                    any_valid = true;
                    channel_frames.push_back(frame);
                } else {
                    // 某路已结束，用黑色填充
                    cv::Mat black(ch->height, ch->width, CV_8UC3, cv::Scalar(0, 0, 0));
                    channel_frames.push_back(black);
                }
            }
            if (!any_valid) break;

            // 逐路检测+跟踪+绘制
            for (size_t i = 0; i < channels.size(); ++i) {
                cv::Mat& frame = channel_frames[i];

                if (!frame.empty() && frame.rows > 1) {
                    // 检测
                    std::vector<cv::Mat> batch = {frame};
                    std::vector<JHDeepCore::DetectionResult> det_results;
                    detector.process(batch, det_results);

                    std::vector<JHDeepCore::Detection> detections;
                    if (!det_results.empty()) {
                        detections = convertDetections(det_results[0]);
                    }

                    // 跟踪
                    std::vector<JHDeepCore::TrackedObject> tracked;
                    channels[i]->tracker.update(detections, frame, tracked);

                    // 绘制
                    drawChannelResult(frame, tracked, channels[i]->name, total_frame_count);
                }

                // 缩放到统一高度
                if (static_cast<int>(frame.rows) != target_height) {
                    cv::Mat resized;
                    double ratio = scale_ratios[i];
                    cv::resize(frame, resized,
                               cv::Size(static_cast<int>(frame.cols * ratio), target_height));
                    scaled_frames.push_back(resized);
                } else {
                    scaled_frames.push_back(frame);
                }
            }

            // 横向拼接
            cv::Mat concat;
            cv::hconcat(scaled_frames, concat);

            writer.write(concat);
            processed_count++;

            // 进度
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
            float progress = static_cast<float>(total_frame_count) / max_frames * 100.0f;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();

            std::cout << "Frame " << total_frame_count << "/" << max_frames
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | Processed: " << processed_count
                      << " | Time: " << frame_ms << "ms"
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }

        // 释放
        for (auto& ch : channels) ch->cap.release();
        writer.release();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        double avg_fps = (total_sec > 0) ? static_cast<double>(processed_count) / total_sec : 0;

        std::cout << "================================" << std::endl;
        std::cout << "Cross-camera tracking completed!" << std::endl;
        std::cout << "Total source frames: " << total_frame_count << std::endl;
        std::cout << "Processed frames: " << processed_count << std::endl;
        std::cout << "Total time: " << total_sec << "s" << std::endl;
        std::cout << "Avg speed: " << std::fixed << std::setprecision(2) << avg_fps << " FPS" << std::endl;
        std::cout << "Output: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
