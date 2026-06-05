#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using json = nlohmann::json;

/// 确保目录存在
static bool ensureDirectoryExists(const std::string& path)
{
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

/// 生成labelme格式的JSON
static json createLabelmeJson(const cv::Mat& image,
                               const std::vector<JHDeepCore::Detection>& detections,
                               const std::string& image_path,
                               const std::string& label_name = "panjuan")
{
    json j;

    // 基本信息
    j["version"] = "5.1.1";
    j["flags"] = {};
    j["shapes"] = json::array();

    // 添加检测结果为矩形标注
    for (const auto& det : detections) {
        json shape;
        shape["label"] = label_name;
        shape["points"] = {
            {det.bbox.x, det.bbox.y},
            {det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height}
        };
        shape["group_id"] = nullptr;
        shape["shape_type"] = "rectangle";
        shape["flags"] = {};

        j["shapes"].push_back(shape);
    }

    // 图像信息
    j["imagePath"] = std::filesystem::path(image_path).filename().string();
    j["imageData"] = nullptr;
    j["imageHeight"] = image.rows;
    j["imageWidth"] = image.cols;

    return j;
}

/// 保存JSON文件
static bool saveJsonFile(const json& j, const std::string& json_path)
{
    try {
        std::ofstream file(json_path);
        if (!file.is_open()) {
            std::cerr << "无法创建JSON文件: " << json_path << std::endl;
            return false;
        }
        file << j.dump(2);  // 缩进2个空格
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "保存JSON失败: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[])
{
    // 默认参数
    std::string model_path = "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.onnx";
    std::string video_path = "/Users/zhanghaining/2026code/jhcv_lib/images/test.mp4";
    std::string label_path = "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.yaml";
    std::string output_dir = "/Users/zhanghaining/2026code/jhcv_lib/result/dataset";
    std::string label_name = "panjuan";
    int skip_frames = 5;  // 每5帧提取一帧
    int device_id = 0;

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
        output_dir = argv[4];
    }
    if (argc >= 6) {
        skip_frames = std::stoi(argv[5]);
    }
    if (argc >= 7) {
        label_name = argv[6];
    }
    if (argc >= 8) {
        device_id = std::stoi(argv[7]);
    }

    // 显示使用信息
    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [video_path] [label_path] [output_dir] [skip_frames] [label_name] [device_id]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/best.onnx video.mp4 labels.txt result/dataset 5 panjuan 0" << std::endl;
        std::cerr << "  model_path: 检测模型路径" << std::endl;
        std::cerr << "  video_path: 输入视频路径" << std::endl;
        std::cerr << "  label_path: 标签配置文件路径" << std::endl;
        std::cerr << "  output_dir: 输出数据集目录" << std::endl;
        std::cerr << "  skip_frames: 每N帧提取一帧（默认5）" << std::endl;
        std::cerr << "  label_name: 标签名称（默认panjuan）" << std::endl;
        std::cerr << "  device_id: 设备ID（默认0）" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== 视频转数据集程序 ===" << std::endl;
        std::cout << "模型路径: " << model_path << std::endl;
        std::cout << "视频路径: " << video_path << std::endl;
        std::cout << "标签路径: " << (label_path.empty() ? "未指定" : label_path) << std::endl;
        std::cout << "输出目录: " << output_dir << std::endl;
        std::cout << "跳帧设置: 每 " << skip_frames << " 帧提取一帧" << std::endl;
        std::cout << "标签名称: " << label_name << std::endl;
        std::cout << "设备ID: " << device_id << std::endl;
        std::cout << "================================" << std::endl;

        // 创建输出目录
        if (!ensureDirectoryExists(output_dir)) {
            std::cerr << "警告: 无法创建输出目录: " << output_dir << std::endl;
        }

        // 初始化检测器
        std::cout << "正在初始化检测器..." << std::endl;
        JHDeepCore::Detector detector(model_path, label_path, device_id);
        std::cout << "检测器初始化完成" << std::endl;
        std::cout << "批处理大小: " << detector.GetBatch() << std::endl;
        std::cout << "输入尺寸: " << detector.GetInputWidth() << "x" << detector.GetInputHeight() << std::endl;

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
        std::cout << "预计提取帧数: " << (total_frames / skip_frames) << std::endl;

        // 处理视频帧
        cv::Mat frame;
        int total_frame_count = 0;
        int saved_frame_count = 0;
        int total_detections = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "开始处理视频..." << std::endl;
        std::cout << "正在提取帧并生成labelme格式的JSON标注..." << std::endl;

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

            // 检测
            std::vector<cv::Mat> images = {frame};
            std::vector<JHDeepCore::DetectionResult> detection_results;
            detector.process(images, detection_results);

            // 转换检测结果
            std::vector<JHDeepCore::Detection> detections;
            if (!detection_results.empty()) {
                for (const auto& det : detection_results[0].detections) {
                    JHDeepCore::Detection d;
                    d.bbox = det.bbox;
                    d.confidence = det.confidence;
                    d.class_id = det.class_id;
                    d.class_name = det.class_name;
                    detections.push_back(d);
                }
            }

            // 生成文件名
            std::string base_name = "frame_" + std::to_string(saved_frame_count + 1);
            std::string image_filename = base_name + ".jpg";
            std::string json_filename = base_name + ".json";

            std::string image_path = output_dir + "/" + image_filename;
            std::string json_path = output_dir + "/" + json_filename;

            // 保存图像
            if (!cv::imwrite(image_path, frame)) {
                std::cerr << "警告: 无法保存图像: " << image_path << std::endl;
                continue;
            }

            // 生成并保存JSON
            json labelme_json = createLabelmeJson(frame, detections, image_path, label_name);
            if (!saveJsonFile(labelme_json, json_path)) {
                std::cerr << "警告: 无法保存JSON: " << json_path << std::endl;
                continue;
            }

            saved_frame_count++;
            total_detections += static_cast<int>(detections.size());

            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);

            // 显示进度
            float progress = static_cast<float>(total_frame_count) / total_frames * 100.0f;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);

            std::cout << "正在处理第 " << total_frame_count << "/" << total_frames << " 帧 "
                      << "(" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " | 已保存: " << saved_frame_count << " 对"
                      << " | 检测目标: " << detections.size()
                      << " | 本帧耗时: " << frame_duration.count() << "ms"
                      << " | 总耗时: " << total_elapsed.count() << "s" << std::endl;
        }

        // 释放资源
        cap.release();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "================================" << std::endl;
        std::cout << "数据集生成完成！" << std::endl;
        std::cout << "原始视频总帧数: " << total_frame_count << std::endl;
        std::cout << "实际保存帧数: " << saved_frame_count << std::endl;
        std::cout << "跳过的帧数: " << (total_frame_count - saved_frame_count) << std::endl;
        std::cout << "总检测目标数: " << total_detections << std::endl;
        std::cout << "总处理时间: " << total_duration.count() << " 秒" << std::endl;
        std::cout << "平均每帧耗时: " << (total_duration.count() > 0 ? total_duration.count() / saved_frame_count : 0) << " 秒" << std::endl;
        std::cout << "输出目录: " << output_dir << std::endl;
        std::cout << "数据格式: labelme JSON + JPG图像" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
