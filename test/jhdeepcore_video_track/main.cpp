#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>

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

/// 根据类别ID生成颜色
static cv::Scalar getColorForClass(int class_id)
{
    return cv::Scalar(
        (class_id * 37) % 256,
        (class_id * 67) % 256,
        (class_id * 97) % 256
    );
}

/// 在帧上绘制跟踪结果
static void drawTrackedObjects(cv::Mat& frame, const std::vector<JHDeepCore::TrackedObject>& tracked_objects)
{
    for (const auto& obj : tracked_objects) {
        cv::Scalar color = getColorForTrackId(obj.track_id);

        // 绘制检测框
        cv::rectangle(frame, obj.bbox, color, 2);

        // 绘制轨迹线
        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::line(frame, obj.trajectory[j], obj.trajectory[j + 1], color, 2, cv::LINE_AA);
            }
            // 轨迹末端画圆点
            cv::circle(frame, obj.trajectory.back(), 4, color, -1, cv::LINE_AA);
        }

        // 绘制ID和类别标签
        std::string label = "ID:" + std::to_string(obj.track_id);
        if (obj.class_id >= 0) {
            label += " cls:" + std::to_string(obj.class_id);
        }
        if (obj.confidence > 0) {
            label += " " + std::to_string(static_cast<int>(obj.confidence * 100)) + "%";
        }

        // 计算文本大小
        int baseLine = 0;
        double fontScale = (frame.cols < 1000) ? 0.45 : 0.6;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);
        int top = std::max(obj.bbox.y, textSize.height + 5);

        // 绘制标签背景
        cv::rectangle(frame,
                     cv::Point(obj.bbox.x, top - textSize.height - 3),
                     cv::Point(obj.bbox.x + textSize.width + 4, top + 2),
                     color, -1);

        // 绘制标签文本
        cv::putText(frame, label,
                   cv::Point(obj.bbox.x + 2, top - 1),
                   cv::FONT_HERSHEY_SIMPLEX, fontScale,
                   cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

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

int main(int argc, char* argv[])
{
    // 默认参数
    std::string model_path = "";  // 需要用户提供
    std::string video_path = "/Users/zhanghaining/2026code/jhcv_lib/images/test.mp4";
    std::string label_path = "";
    std::string output_path = "result/video_track_result.avi";
    int device_id = 0;
    int skip_frames = 3;  // 默认每3帧检测一次，即跳过2帧

    // 解析命令行参数
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        video_path = argv[2];
    }
    if (argc >= 4) {
        label_path = argv[3];
    }
    if (argc >= 5) {
        output_path = argv[4];
    }
    if (argc >= 6) {
        device_id = std::stoi(argv[5]);
    }
    if (argc >= 7) {
        skip_frames = std::stoi(argv[6]);
    }

    // 显示使用信息
    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [video_path] [label_path] [output_path] [device_id] [skip_frames]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/yolo.onnx video.mp4 labels.txt result/output.avi 0 3" << std::endl;
        std::cerr << "  skip_frames: 跳帧间隔（默认3，表示每3帧处理1帧，跳过2帧）" << std::endl;
        std::cerr << "                输出视频将只有原始视频的 1/skip_frames 帧数" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== 视频检测与跟踪测试程序 ===" << std::endl;
        std::cout << "模型路径: " << model_path << std::endl;
        std::cout << "视频路径: " << video_path << std::endl;
        std::cout << "标签路径: " << (label_path.empty() ? "未指定" : label_path) << std::endl;
        std::cout << "输出路径: " << output_path << std::endl;
        std::cout << "设备ID: " << device_id << std::endl;
        std::cout << "跳帧设置: 每 " << skip_frames << " 帧检测一次（跳过 " << (skip_frames - 1) << " 帧）" << std::endl;
        std::cout << "================================" << std::endl;

        // 创建结果目录
        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) {
            ensureDirectoryExists(result_dir);
        }

        // 初始化检测器
        std::cout << "正在初始化检测器..." << std::endl;
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "检测器初始化完成" << std::endl;
        std::cout << "批处理大小: " << detector.GetBatch() << std::endl;
        std::cout << "输入尺寸: " << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

        // 初始化跟踪器
        std::cout << "正在初始化跟踪器..." << std::endl;
        JHDeepCore::TrackerConfig tracker_config;
        tracker_config.tracker_type = JHDeepCore::TrackerType::ByteTrack;
        tracker_config.distance_type = JHDeepCore::TrackDistanceType::IoU;
        tracker_config.distance_threshold = 0.6f;
        tracker_config.bytetrack_track_thresh = 0.5f;
        tracker_config.bytetrack_high_thresh = 0.5f;
        tracker_config.bytetrack_match_thresh = 0.8f;

        // 打开视频文件
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "错误: 无法打开视频文件: " << video_path << std::endl;
            return 1;
        }

        // 获取视频信息
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "视频信息: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
        std::cout << "总帧数: " << total_frames << std::endl;

        // 使用视频实际FPS初始化跟踪器
        JHDeepCore::Tracker tracker(tracker_config, static_cast<float>(fps));

        // 根据输出文件扩展名选择编码器
        std::string ext = output_path.substr(output_path.find_last_of('.'));
        int fourcc = 0;

        if (ext == ".avi") {
            // AVI格式，使用XVID编码，支持实时播放
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            std::cout << "使用AVI格式 (XVID编码)，支持实时播放" << std::endl;
        } else if (ext == ".mkv") {
            // MKV格式，使用H264编码
            fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
            std::cout << "使用MKV格式 (H264编码)" << std::endl;
        } else {
            // 默认使用AVI格式以确保实时播放
            std::string new_output = output_path.substr(0, output_path.find_last_of('.')) + ".avi";
            output_path = new_output;
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            std::cout << "自动切换到AVI格式以确保实时播放能力" << std::endl;
            std::cout << "输出路径更改为: " << output_path << std::endl;
        }

        // 创建视频写入器
        cv::VideoWriter writer;
        writer.open(output_path, fourcc, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "错误: 无法创建输出视频文件: " << output_path << std::endl;
            return 1;
        }

        std::cout << "视频编码器已初始化，可以实时播放输出文件" << std::endl;

        // 处理视频帧
        cv::Mat frame;
        int total_frame_count = 0;      // 视频总帧数计数
        int processed_frame_count = 0; // 实际处理的帧数
        int detection_count = 0;
        int total_tracked = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "开始处理视频..." << std::endl;
        std::cout << "正在使用OpenCV读取视频帧，每" << skip_frames << "帧处理一次..." << std::endl;

        while (true) {
            // 读取一帧
            bool has_frame = cap.read(frame);
            if (!has_frame) break;

            total_frame_count++;

            // 判断是否需要处理这一帧
            if (total_frame_count % skip_frames != 0) {
                continue;  // 跳过这一帧
            }

            auto frame_start = std::chrono::high_resolution_clock::now();

            // 处理当前帧：检测
            std::vector<cv::Mat> images = {frame};
            std::vector<JHDeepCore::DetectionResult> detection_results;
            detector.process(images, detection_results);

            // 转换检测结果
            std::vector<JHDeepCore::Detection> detections;
            if (!detection_results.empty()) {
                detections = convertDetections(detection_results[0]);
            }

            // 处理当前帧：跟踪
            std::vector<JHDeepCore::TrackedObject> tracked_objects;
            tracker.update(detections, frame, tracked_objects);

            // 绘制跟踪结果
            cv::Mat output_frame = frame.clone();
            drawTrackedObjects(output_frame, tracked_objects);

            // 写入输出视频
            writer.write(output_frame);

            processed_frame_count++;
            detection_count++;
            total_tracked = std::max(total_tracked, static_cast<int>(tracked_objects.size()));

            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);

            // 显示进度
            float progress = static_cast<float>(total_frame_count) / total_frames * 100.0f;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);

            std::cout << "正在处理第 " << total_frame_count << "/" << total_frames << " 帧 "
                      << "(" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | 已处理: " << processed_frame_count << " 帧"
                      << " | 检测目标: " << detections.size()
                      << " | 跟踪目标: " << tracked_objects.size()
                      << " | 本帧耗时: " << frame_duration.count() << "ms"
                      << " | 总耗时: " << total_elapsed.count() << "s" << std::endl;
        }

        // 释放资源
        cap.release();
        writer.release();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        double avg_fps = static_cast<double>(processed_frame_count) / total_duration.count();

        std::cout << "================================" << std::endl;
        std::cout << "视频处理完成！" << std::endl;
        std::cout << "原始视频总帧数: " << total_frame_count << std::endl;
        std::cout << "实际处理帧数: " << processed_frame_count << std::endl;
        std::cout << "跳过的帧数: " << (total_frame_count - processed_frame_count) << std::endl;
        std::cout << "输出视频帧数: " << processed_frame_count << " (原始视频的 1/" << skip_frames << ")" << std::endl;
        std::cout << "同时出现的最大目标数: " << total_tracked << std::endl;
        std::cout << "总处理时间: " << total_duration.count() << " 秒" << std::endl;
        std::cout << "平均处理速度: " << std::fixed << std::setprecision(2) << avg_fps << " FPS" << std::endl;
        std::cout << "输出视频路径: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
