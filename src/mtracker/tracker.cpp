#include "JHDeepCore.h"
#include "BaseTracker.h"
#include "TrackerSettings.h"
#include "defines.h"
#include <stdexcept>
#include <chrono>

namespace JHDeepCore {

class TrackerPrivate {
public:
    TrackerPrivate(const TrackerConfig &config, float fps) {
        TrackerSettings settings;

        // 跟踪器类型
        settings.m_tracker = (config.tracker_type == TrackerType::ByteTrack)
                                 ? tracking::ByteTrack
                                 : tracking::UniversalTracker;

        // 距离度量
        switch (config.distance_type) {
        case TrackDistanceType::Centers:
            settings.SetDistance(tracking::DistCenters);
            break;
        case TrackDistanceType::Rects:
            settings.SetDistance(tracking::DistRects);
            break;
        case TrackDistanceType::IoU:
        default:
            settings.SetDistance(tracking::DistJaccard);
            break;
        }

        // 通用参数
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterCenter;
        settings.m_lostTrackType = tracking::TrackNone;
        settings.m_matchType = tracking::MatchLAPJV;
        settings.m_dt = config.kalman_dt;
        settings.m_accelNoiseMag = config.accel_noise;
        settings.m_distThres = config.distance_threshold;
        settings.m_maximumAllowedLostTime = config.max_lost_time;
        settings.m_maxTraceLength = config.max_trace_length;

        // ByteTrack 参数
        settings.m_byteTrackSettings.m_trackBuffer = config.bytetrack_track_buffer;
        settings.m_byteTrackSettings.m_trackThresh = config.bytetrack_track_thresh;
        settings.m_byteTrackSettings.m_highThresh = config.bytetrack_high_thresh;
        settings.m_byteTrackSettings.m_matchThresh = config.bytetrack_match_thresh;

        tracker_ = BaseTracker::CreateTracker(settings, fps);
        if (!tracker_) {
            throw std::runtime_error("Failed to create tracker");
        }
    }

    void update(const std::vector<Detection> &detections,
                const cv::Mat &frame,
                std::vector<TrackedObject> &tracked_objects) {
        // 将 JHDeepCore::Detection 转换为内部 CRegion
        regions_t regions;
        regions.reserve(detections.size());
        for (const auto &det : detections) {
            regions.emplace_back(det.bbox, static_cast<objtype_t>(det.class_id), det.confidence);
        }

        auto frameTime = std::chrono::system_clock::now();
        tracker_->UpdateMat(regions, frame, frameTime);

        // 获取跟踪结果并转换
        std::vector<TrackingObject> internalTracks;
        tracker_->GetTracks(internalTracks);

        tracked_objects.clear();
        tracked_objects.reserve(internalTracks.size());
        for (const auto &t : internalTracks) {
            TrackedObject obj;
            obj.track_id = t.m_ID.m_val;
            obj.bbox = t.m_rrect.boundingRect();
            obj.class_id = static_cast<int>(t.m_type);
            obj.confidence = t.m_confidence;
            obj.trajectory = t.GetTrajectory();
            obj.is_stable = t.IsRobust(5, 0.5f, cv::Size2f(0.1f, 8.0f));
            tracked_objects.push_back(obj);
        }
    }

    void get_removed_ids(std::vector<size_t> &removed_ids) {
        std::vector<track_id_t> internalIds;
        tracker_->GetRemovedTracks(internalIds);
        removed_ids.clear();
        removed_ids.reserve(internalIds.size());
        for (const auto &id : internalIds) {
            removed_ids.push_back(id.m_val);
        }
    }

private:
    std::unique_ptr<BaseTracker> tracker_;
};

Tracker::Tracker(const TrackerConfig &config, float fps)
    : m_pHandle(std::make_shared<TrackerPrivate>(config, fps)) {}

Tracker::~Tracker() = default;

void Tracker::update(const std::vector<Detection> &detections,
                     const cv::Mat &frame,
                     std::vector<TrackedObject> &tracked_objects) {
    m_pHandle->update(detections, frame, tracked_objects);
}

void Tracker::get_removed_ids(std::vector<size_t> &removed_ids) {
    m_pHandle->get_removed_ids(removed_ids);
}

} // namespace JHDeepCore
