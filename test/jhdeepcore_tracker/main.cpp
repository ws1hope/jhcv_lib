#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

static bool ensureDirectoryExists(const std::string &path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

/// 根据跟踪 ID 生成固定颜色
static cv::Scalar getColor(size_t track_id) {
    static const cv::Scalar colors[] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 0, 255}, {255, 128, 0},
    };
    return colors[track_id % 8];
}

/// 模拟 6 个点目标的检测结果，含交叉、相遇、消失等场景
static void generate_detections(int frame_id, int w, int h,
                                std::vector<JHDeepCore::Detection> &dets) {
    dets.clear();
    float t = static_cast<float>(frame_id);
    auto add = [&](int cid, const std::string &name, float conf, cv::Point2f center, cv::Size2f size) {
        if (center.x - size.width / 2 < 0 || center.x + size.width / 2 > w ||
            center.y - size.height / 2 < 0 || center.y + size.height / 2 > h)
            return;
        JHDeepCore::Detection d;
        d.class_id = cid;
        d.class_name = name;
        d.confidence = conf;
        d.bbox = cv::Rect(
            static_cast<int>(center.x - size.width / 2),
            static_cast<int>(center.y - size.height / 2),
            static_cast<int>(size.width),
            static_cast<int>(size.height));
        dets.push_back(d);
    };

    // P0: 匀速向右（人）
    add(0, "person", 0.92f, {30 + t * 8, 120}, {30, 60});

    // P1: 匀速向下（车）
    add(1, "car", 0.88f, {200, 30 + t * 7}, {50, 30});

    // P2: 左上到右下对角线运动（卡车）
    add(2, "truck", 0.85f, {20 + t * 6, 20 + t * 5}, {60, 40});

    // P3: 从右向左运动（人），与 P0 方向相反，中间会交叉
    add(0, "person", 0.80f, {w - 30 - t * 7, 130}, {28, 55});

    // P4: 圆周运动（自行车）
    {
        float cx = 320, cy = 300, r = 100.f;
        float angle = t * 0.1f;
        add(3, "bicycle", 0.78f, {cx + r * cosf(angle), cy + r * sinf(angle)}, {35, 35});
    }

    // P5: 前 40 帧存在，之后消失（模拟目标离开画面）
    if (frame_id < 40) {
        add(1, "car", 0.82f, {500 - t * 4, 400 - t * 3}, {45, 30});
    }
}

/// 在帧上绘制跟踪结果
static void draw_tracked(cv::Mat &frame, const std::vector<JHDeepCore::TrackedObject> &objects) {
    for (const auto &obj : objects) {
        cv::Scalar color = getColor(obj.track_id);

        // 画检测框
        cv::rectangle(frame, obj.bbox, color, 2);

        // 画轨迹线
        if (obj.trajectory.size() > 1) {
            for (size_t j = 0; j < obj.trajectory.size() - 1; ++j) {
                cv::line(frame, obj.trajectory[j], obj.trajectory[j + 1], color, 2, cv::LINE_AA);
            }
            // 轨迹末端画圆点
            cv::circle(frame, obj.trajectory.back(), 4, color, -1, cv::LINE_AA);
        }

        // 画 ID 和类别标签
        std::string label = "ID" + std::to_string(obj.track_id) + " " + std::to_string(obj.class_id);
        int baseline = 0;
        double fontScale = (frame.cols < 1000) ? 0.45 : 0.6;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
        int top = std::max(obj.bbox.y, textSize.height + 5);
        cv::rectangle(frame,
                       cv::Point(obj.bbox.x, top - textSize.height - 3),
                       cv::Point(obj.bbox.x + textSize.width + 4, top + 2),
                       color, -1);
        cv::putText(frame, label,
                    cv::Point(obj.bbox.x + 2, top - 1),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

int main() {
    try {
        ensureDirectoryExists("result");

        JHDeepCore::TrackerConfig config;
        config.tracker_type = JHDeepCore::TrackerType::ByteTrack;
        config.distance_type = JHDeepCore::TrackDistanceType::IoU;
        config.distance_threshold = 0.7f;

        double fps = 30.0;
        int total_frames = 90;
        int width = 960;
        int height = 540;

        JHDeepCore::Tracker tracker(config, static_cast<float>(fps));

        // 创建视频写入器
        cv::VideoWriter writer("result/tracker_test.mp4",
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Failed to create video writer" << std::endl;
            return 1;
        }

        std::cout << "=== Tracker Video Test ===" << std::endl;
        std::cout << "Simulating " << total_frames << " frames, 6 objects with crossings..." << std::endl;

        for (int i = 0; i < total_frames; ++i) {
            // 深灰背景 + 网格
            cv::Mat frame(height, width, CV_8UC3, cv::Scalar(40, 40, 40));
            for (int x = 0; x < width; x += 80)
                cv::line(frame, cv::Point(x, 0), cv::Point(x, height), cv::Scalar(60, 60, 60), 1);
            for (int y = 0; y < height; y += 80)
                cv::line(frame, cv::Point(0, y), cv::Point(width, y), cv::Scalar(60, 60, 60), 1);

            // 帧号水印
            std::string frameText = "Frame " + std::to_string(i);
            cv::putText(frame, frameText, cv::Point(width - 160, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);

            // 生成检测并跟踪
            std::vector<JHDeepCore::Detection> detections;
            generate_detections(i, width, height, detections);

            std::vector<JHDeepCore::TrackedObject> tracked;
            tracker.update(detections, frame, tracked);

            // 绘制跟踪结果
            draw_tracked(frame, tracked);

            writer << frame;

            // 控制台输出
            if (i % 15 == 0 || i == total_frames - 1) {
                std::cout << "Frame " << i << ": " << tracked.size() << " tracked" << std::endl;
                for (const auto &obj : tracked) {
                    std::cout << "  ID=" << obj.track_id
                              << " cls=" << obj.class_id
                              << " bbox=(" << obj.bbox.x << "," << obj.bbox.y
                              << "," << obj.bbox.width << "," << obj.bbox.height << ")"
                              << " traj=" << obj.trajectory.size()
                              << std::endl;
                }
            }
        }

        writer.release();
        std::cout << "\nVideo saved to: result/tracker_test.mp4" << std::endl;
        std::cout << "=== Done ===" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
