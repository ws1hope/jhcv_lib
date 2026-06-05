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

int main(int argc, char* argv[])
{
    std::string model_path = "";
    std::string video_path = "/Users/zhanghaining/2026code/jhcv_lib/images/td_video/td1.mp4";
    std::string label_path = "";
    std::string output_path = "result/homography_result.avi";
    int device_id = 0;
    int skip_frames = 3;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) video_path = argv[2];
    if (argc >= 4) label_path = argv[3];
    if (argc >= 5) output_path = argv[4];
    if (argc >= 6) device_id = std::stoi(argv[5]);
    if (argc >= 7) skip_frames = std::stoi(argv[6]);

    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> [video_path] [label_path] [output_path] [device_id] [skip_frames]"
                  << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Homography Projection Test ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Video: " << video_path << std::endl;
        std::cout << "Output: " << output_path << std::endl;

        // 创建结果目录
        std::string result_dir = output_path.substr(0, output_path.find_last_of("/\\"));
        if (!result_dir.empty()) ensureDirectoryExists(result_dir);

        // 初始化检测器
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "Detector ready: " << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

        // 构造单应矩阵：将原图坐标映射到白图坐标
        // 原图四点 -> 白图四点
        std::vector<PointPair> pairs = {
            {{1692, 153},  {200, 836}},
            {{1892, 231},  {200, 706}},
            {{818,  1439}, {668, 706}},
            {{292,  1175}, {668, 836}},
        };

        Homography homo;
        cv::Mat H = homo.compute(pairs);
        if (H.empty()) {
            std::cerr << "Error: Failed to compute homography matrix" << std::endl;
            return 1;
        }
        std::cout << "Homography matrix:" << std::endl << H << std::endl;

        // 打开视频
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video: " << video_path << std::endl;
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int src_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int src_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "Video: " << src_width << "x" << src_height << " @ " << fps << " FPS, "
                  << total_frames << " frames" << std::endl;

        // 输出画布：上原图(2560x1440)，左下白底(1000x500)，右下黑色
        // 总尺寸 2560x1940
        int map_width = 2000;
        int map_height = 1000;
        int canvas_width = src_width;
        int canvas_height = src_height + map_height;

        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                               fps, cv::Size(canvas_width, canvas_height));
        if (!writer.isOpened()) {
            std::cerr << "Error: Cannot create output video" << std::endl;
            return 1;
        }

        std::cout << "Canvas output: " << canvas_width << "x" << canvas_height << std::endl;
        std::cout << "Map region: " << map_width << "x" << map_height << " at (0, " << src_height << ")" << std::endl;

        // 主循环
        cv::Mat frame;
        int total_count = 0;
        int processed_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            if (!cap.read(frame) || frame.empty()) break;
            total_count++;

            if (total_count % skip_frames != 0) continue;

            // 检测
            std::vector<cv::Mat> batch = {frame};
            std::vector<JHDeepCore::DetectionResult> det_results;
            detector.process(batch, det_results);

            // 创建画布：全黑
            cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(0, 0, 0));

            // 原图放到上方
            frame.copyTo(canvas(cv::Rect(0, 0, src_width, src_height)));

            // 左下角白底区域
            cv::Mat white_roi = canvas(cv::Rect(0, src_height, map_width, map_height));
            white_roi.setTo(cv::Scalar(255, 255, 255));

            // 检测
            if (!det_results.empty()) {
                for (const auto& det : det_results[0].detections) {
                    // 原图区域绘制检测框（canvas左上角）
                    cv::Rect bbox_shifted = det.bbox;
                    cv::rectangle(canvas, bbox_shifted, cv::Scalar(0, 255, 0), 2);

                    // 计算中心点
                    cv::Point2f center(det.bbox.x + det.bbox.width / 2.0f,
                                       det.bbox.y + det.bbox.height / 2.0f);

                    // 原图上标记中心点
                    cv::circle(canvas, center, 6, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

                    // 通过单应矩阵映射到白底区域（坐标基于1000x500）
                    cv::Point2f mapped = homo.project_point(center);
                    // 映射到canvas上的实际位置（左下角白底区域偏移）
                    cv::Point canvas_mapped(static_cast<int>(mapped.x),
                                            static_cast<int>(mapped.y) + src_height);
                    cv::circle(canvas, canvas_mapped, 8, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

                    // 标签
                    std::ostringstream oss;
                    oss << det.class_id << "_" << std::fixed << std::setprecision(2) << det.confidence;
                    int baseLine = 0;
                    double labelScale = 0.6;
                    cv::Size textSize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, labelScale, 2, &baseLine);
                    int top = std::max(static_cast<int>(det.bbox.y), textSize.height + 5);
                    cv::rectangle(canvas,
                                  cv::Point(det.bbox.x, top - textSize.height - 3),
                                  cv::Point(det.bbox.x + textSize.width + 4, top + 2),
                                  cv::Scalar(0, 255, 0), -1);
                    cv::putText(canvas, oss.str(), cv::Point(det.bbox.x + 2, top - 1),
                                cv::FONT_HERSHEY_SIMPLEX, labelScale, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                }
            }

            writer.write(canvas);
            processed_count++;

            // 进度
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            float progress = static_cast<float>(total_count) / total_frames * 100.0f;
            int det_count = det_results.empty() ? 0 : static_cast<int>(det_results[0].detections.size());

            std::cout << "Frame " << total_count << "/" << total_frames
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | Detections: " << det_count
                      << " | Processed: " << processed_count
                      << " | Elapsed: " << elapsed << "s" << std::endl;
        }

        cap.release();
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
