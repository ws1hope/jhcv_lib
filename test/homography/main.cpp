#include "JHDeepCore.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

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

/// 在指定区域绘制检测框、中心点和标签
static void drawDetections(cv::Mat& canvas, int x_offset,
                           const std::vector<JHDeepCore::Detection>& detections,
                           Homography& homo, int top_height,
                           const cv::Scalar& map_point_color)
{
    for (const auto& det : detections) {
        cv::Rect bbox_shifted = det.bbox;
        bbox_shifted.x += x_offset;
        cv::rectangle(canvas, bbox_shifted, cv::Scalar(0, 255, 0), 2);

        cv::Point2f center(det.bbox.x + det.bbox.width / 2.0f,
                           det.bbox.y + det.bbox.height / 2.0f);

        cv::circle(canvas, cv::Point(static_cast<int>(center.x) + x_offset,
                                     static_cast<int>(center.y)),
                   6, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

        cv::Point2f mapped = homo.project_point(center);
        cv::Point canvas_mapped(static_cast<int>(mapped.x),
                                static_cast<int>(mapped.y) + top_height);
        cv::circle(canvas, canvas_mapped, 8, map_point_color, -1, cv::LINE_AA);

        std::ostringstream oss;
        oss << det.class_id << "_" << std::fixed << std::setprecision(2) << det.confidence;
        int baseLine = 0;
        double labelScale = 0.6;
        cv::Size textSize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, labelScale, 2, &baseLine);
        int top = std::max(static_cast<int>(det.bbox.y), textSize.height + 5);
        cv::rectangle(canvas,
                      cv::Point(bbox_shifted.x, top - textSize.height - 3),
                      cv::Point(bbox_shifted.x + textSize.width + 4, top + 2),
                      cv::Scalar(0, 255, 0), -1);
        cv::putText(canvas, oss.str(), cv::Point(bbox_shifted.x + 2, top - 1),
                    cv::FONT_HERSHEY_SIMPLEX, labelScale, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}

/// 单个视频通道
struct VideoChannel {
    std::string path;
    cv::VideoCapture cap;
    Homography homo;
    cv::Scalar map_color;
    int width = 0;
    int height = 0;
    int x_offset = 0;
};

int main(int argc, char* argv[])
{
    std::string model_path = "";
    std::string video1_path = "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td1.mp4";
    std::string video2_path = "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td2.mp4";
    std::string video3_path = "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td3.mp4";
    std::string label_path = "";
    std::string output_path = "result/homography_result.avi";
    int device_id = 0;
    int skip_frames = 3;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) video1_path = argv[2];
    if (argc >= 4) video2_path = argv[3];
    if (argc >= 5) video3_path = argv[4];
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
        std::cout << "=== Multi-Video Homography Projection Test ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Video 1: " << video1_path << std::endl;
        std::cout << "Video 2: " << video2_path << std::endl;
        std::cout << "Video 3: " << video3_path << std::endl;
        std::cout << "Output: " << output_path << std::endl;

        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready: " << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

        // 各视频单应矩阵点对（映射坐标已翻倍，对应4000x2000白底）
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

        // 初始化三个通道
        std::vector<std::unique_ptr<VideoChannel>> channels(3);
        std::vector<std::string> video_paths = {video1_path, video2_path, video3_path};
        std::vector<cv::Scalar> map_colors = {
            cv::Scalar(0, 0, 255),     // 红色
            cv::Scalar(255, 0, 0),     // 蓝色
            cv::Scalar(0, 255, 0),     // 绿色
        };
        std::vector<std::vector<PointPair>> all_pairs = {pairs1, pairs2, pairs3};

        for (size_t i = 0; i < channels.size(); ++i) {
            auto ch = std::make_unique<VideoChannel>();
            ch->path = video_paths[i];
            ch->map_color = map_colors[i];
            ch->cap.open(ch->path);
            if (!ch->cap.isOpened()) {
                std::cerr << "Error: Cannot open video " << (i + 1) << ": " << ch->path << std::endl;
                return 1;
            }
            ch->width = static_cast<int>(ch->cap.get(cv::CAP_PROP_FRAME_WIDTH));
            ch->height = static_cast<int>(ch->cap.get(cv::CAP_PROP_FRAME_HEIGHT));

            cv::Mat H = ch->homo.compute(all_pairs[i]);
            if (H.empty()) {
                std::cerr << "Error: Failed to compute homography for video " << (i + 1) << std::endl;
                return 1;
            }
            std::cout << "Video " << (i + 1) << ": " << ch->width << "x" << ch->height
                      << " | Homography computed" << std::endl;
            channels[i] = std::move(ch);
        }

        // 计算布局
        int top_height = 0;
        int top_width = 0;
        int total_frames = INT_MAX;
        for (size_t i = 0; i < channels.size(); ++i) {
            top_height = std::max(top_height, channels[i]->height);
            channels[i]->x_offset = top_width;
            top_width += channels[i]->width;
            total_frames = std::min(total_frames,
                static_cast<int>(channels[i]->cap.get(cv::CAP_PROP_FRAME_COUNT)));
        }

        double fps = channels[0]->cap.get(cv::CAP_PROP_FPS);

        int map_width = 7500;
        int map_height = 1000;
        int canvas_width = top_width;
        int canvas_height = top_height + map_height;

        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                               fps, cv::Size(canvas_width, canvas_height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video" << std::endl;
            return 1;
        }
        std::cout << "Canvas: " << canvas_width << "x" << canvas_height << std::endl;
        std::cout << "Total frames: " << total_frames << " @ " << fps << " FPS" << std::endl;

        // 主循环
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

            // 创建画布
            cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(0, 0, 0));

            // 放置三路视频
            for (size_t i = 0; i < channels.size(); ++i) {
                frames[i].copyTo(canvas(cv::Rect(channels[i]->x_offset, 0,
                                                  channels[i]->width, channels[i]->height)));
            }

            // 左下白底
            cv::Mat white_roi = canvas(cv::Rect(0, top_height, map_width, map_height));
            white_roi.setTo(cv::Scalar(255, 255, 255));

            // 逐路检测+绘制
            for (size_t i = 0; i < channels.size(); ++i) {
                std::vector<cv::Mat> batch = {frames[i]};
                std::vector<JHDeepCore::DetectionResult> det_results;
                detector.process(batch, det_results);

                std::vector<JHDeepCore::Detection> dets;
                if (!det_results.empty()) dets = det_results[0].detections;
                drawDetections(canvas, channels[i]->x_offset, dets,
                               channels[i]->homo, top_height, channels[i]->map_color);
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
        std::cout << "================================" << std::endl;
        std::cout << "Completed! Processed " << processed_count << " frames in " << total_sec << "s" << std::endl;
        std::cout << "Output: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
