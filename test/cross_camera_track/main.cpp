#include <iostream>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include <chrono>
#include <iomanip>
#include <map>
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
    CameraId camera_id = 0;
    cv::Scalar map_color;
    int width = 0;
    int height = 0;
    int x_offset = 0;
    double fps = 0;
    int total_frames = 0;

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

struct GlobalTargetVisualState {
    cv::Point2f point;
    CameraId camera_id = 0;
    bool initialized = false;
    bool transitioning = false;
    size_t missing_updates = 0;
};

/// Draw local tracking diagnostics in the camera view.
static void drawTrackedObjects(cv::Mat& canvas, CameraChannel& ch,
                               const std::vector<JHDeepCore::CrossCameraTrackedObject>& tracked_objects)
{
    for (const auto& cross_obj : tracked_objects) {
        if (cross_obj.camera_id != ch.camera_id) {
            continue;
        }

        const auto& obj = cross_obj.local_track;
        cv::Scalar color = getColorForTrackId(cross_obj.target_id);

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

        // Large target ID shared by adjacent cameras.
        std::string label = std::to_string(cross_obj.target_id);
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

        // Small local ID remains visible for tracker diagnostics.
        cv::putText(canvas, "L:" + std::to_string(obj.track_id),
                    cv::Point(bbox_shifted.x, top + 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);
    }
}

/// Draw one EMA-smoothed map position for each global target.
static void drawGlobalTargets(
    cv::Mat& canvas,
    const std::vector<JHDeepCore::CrossCameraTrackedObject>& tracked_objects,
    int top_height,
    std::map<TargetId, GlobalTargetVisualState>& visual_states)
{
    constexpr float ema_alpha = 0.2f;
    constexpr float transition_stop_distance = 2.0f;
    constexpr size_t state_retention_updates = 30;

    // Camera IDs describe the physical order, so the later camera wins while
    // both local tracks are visible in an overlap area.
    std::map<TargetId, const CrossCameraTrackedObject*> selected_objects;
    for (const auto& object : tracked_objects) {
        auto selected = selected_objects.find(object.target_id);
        if (selected == selected_objects.end() ||
            object.camera_id > selected->second->camera_id) {
            selected_objects[object.target_id] = &object;
        }
    }

    for (auto& [target_id, state] : visual_states) {
        (void)target_id;
        ++state.missing_updates;
    }

    for (const auto& [target_id, object] : selected_objects) {
        auto& state = visual_states[target_id];
        const cv::Point2f desired_point = object->mapped_point;
        state.missing_updates = 0;

        if (!state.initialized) {
            state.point = desired_point;
            state.camera_id = object->camera_id;
            state.initialized = true;
        } else {
            if (state.camera_id != object->camera_id) {
                state.camera_id = object->camera_id;
                state.transitioning = true;
            }

            if (state.transitioning) {
                state.point =
                    state.point * (1.0f - ema_alpha) +
                    desired_point * ema_alpha;
                if (cv::norm(state.point - desired_point) <=
                    transition_stop_distance) {
                    state.point = desired_point;
                    state.transitioning = false;
                }
            } else {
                state.point = desired_point;
            }
        }

        const cv::Point canvas_point(
            cvRound(state.point.x), cvRound(state.point.y) + top_height);
        const cv::Scalar color = getColorForTrackId(target_id);
        cv::circle(canvas, canvas_point, 8, color, -1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(target_id),
                    cv::Point(canvas_point.x - 15, canvas_point.y - 16),
                    cv::FONT_HERSHEY_SIMPLEX, 1.5, color, 4, cv::LINE_AA);
    }

    for (auto state = visual_states.begin(); state != visual_states.end();) {
        if (state->second.missing_updates > state_retention_updates) {
            state = visual_states.erase(state);
        } else {
            ++state;
        }
    }
}

int main(int argc, char* argv[])
{
    std::string model_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.onnx";
    std::string label_path =
        "/Users/zhanghaining/2026code/jhcv_lib/models/panjuan_det/best.yaml";
    std::string output_path = "result/cross_camera_result.avi";
    int device_id = -1;
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
            {{951,  88},   {4800, 702}},
            {{1190, 0},    {4800, 836}},
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
                video_paths[i], "Cam" + std::to_string(i + 1), i + 1);
            ch->map_color = map_colors[i];

            std::cout << "Cam" << (i + 1) << ": " << ch->width << "x" << ch->height
                      << " @ " << ch->fps << " FPS, " << ch->total_frames << " frames"
                      << std::endl;
            if (i == 0) ref_fps = static_cast<float>(ch->fps);
            channels.push_back(std::move(ch));
        }

        CrossCameraTrackerConfig cross_config;
        cross_config.tracker_config = createTrackerConfig();
        cross_config.enable_log = true;
        cross_config.log_directory = "logs";
        for (size_t i = 0; i < channels.size(); ++i) {
            CrossCameraChannelConfig channel_config;
            channel_config.camera_id = channels[i]->camera_id;
            channel_config.tracker_fps =
                static_cast<float>(channels[i]->fps) /
                static_cast<float>(std::max(skip_frames, 1));
            for (const auto& pair : all_pairs[i]) {
                channel_config.calibration_points.push_back(
                    {pair.first, pair.second});
            }
            cross_config.channels.push_back(std::move(channel_config));
        }
        cross_config.links = {
            {channels[0]->camera_id, channels[1]->camera_id, 200.0f},
            {channels[1]->camera_id, channels[2]->camera_id, 200.0f},
        };
        CrossCameraTracker cross_tracker(cross_config);

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
        std::map<TargetId, GlobalTargetVisualState> global_visual_states;
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

            // Run one image at a time because the model may only support batch size 1.
            std::vector<CrossCameraFrameInput> cross_inputs;
            cross_inputs.reserve(channels.size());
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

            std::vector<CrossCameraTrackedObject> tracked_objects;
            cross_tracker.update(cross_inputs, tracked_objects);
            for (auto& channel : channels) {
                drawTrackedObjects(canvas, *channel, tracked_objects);
            }
            drawGlobalTargets(
                canvas, tracked_objects, top_height, global_visual_states);

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
