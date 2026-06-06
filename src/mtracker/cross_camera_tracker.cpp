#include "JHDeepCore.h"

#include "ShortPathCalculator.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace JHDeepCore {
namespace {

struct LocalTrackKey {
    CameraId camera_id = 0;
    size_t local_id = 0;

    bool operator<(const LocalTrackKey &other) const
    {
        return std::tie(camera_id, local_id) <
               std::tie(other.camera_id, other.local_id);
    }
};

class FileLogger {
  public:
    FileLogger(bool enabled, const std::string &directory)
        : enabled_(enabled), directory_(directory)
    {
        if (!enabled_) {
            return;
        }

        std::error_code error;
        std::filesystem::create_directories(directory_, error);
        if (error) {
            throw std::runtime_error("Cannot create log directory: " +
                                     directory_ + ": " + error.message());
        }

        base_name_ = "cross_camera_tracker_" + currentTimeForFile();
        openFile();
    }

    void info(const std::string &event, const std::string &message)
    {
        write("INFO", event, message, false);
    }

    void debug(const std::string &event, const std::string &message)
    {
        write("DEBUG", event, message, false);
    }

    void warning(const std::string &event, const std::string &message)
    {
        write("WARNING", event, message, true);
    }

    void error(const std::string &event, const std::string &message)
    {
        write("ERROR", event, message, true);
    }

  private:
    static constexpr std::streamoff kMaxFileSize =
        static_cast<std::streamoff>(100) * 1024 * 1024;

    bool enabled_ = false;
    std::string directory_;
    std::string base_name_;
    size_t part_ = 0;
    std::ofstream stream_;
    std::mutex mutex_;

    static std::tm localTime(std::time_t value)
    {
        std::tm result{};
#ifdef _WIN32
        localtime_s(&result, &value);
#else
        localtime_r(&value, &result);
#endif
        return result;
    }

    static std::string currentTimeForFile()
    {
        const auto now = std::chrono::system_clock::now();
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
        const std::time_t value = std::chrono::system_clock::to_time_t(now);
        const std::tm time = localTime(value);

        std::ostringstream out;
        out << std::put_time(&time, "%Y%m%d_%H%M%S") << '_'
            << std::setfill('0') << std::setw(3) << milliseconds.count();
        return out.str();
    }

    static std::string currentTimeForLine()
    {
        const auto now = std::chrono::system_clock::now();
        const auto milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
        const std::time_t value = std::chrono::system_clock::to_time_t(now);
        const std::tm time = localTime(value);

        std::ostringstream out;
        out << std::put_time(&time, "%Y-%m-%d %H:%M:%S") << '.'
            << std::setfill('0') << std::setw(3) << milliseconds.count();
        return out.str();
    }

    std::filesystem::path currentPath() const
    {
        std::string filename = base_name_;
        if (part_ > 0) {
            filename += "." + std::to_string(part_);
        }
        filename += ".log";
        return std::filesystem::path(directory_) / filename;
    }

    void openFile()
    {
        stream_.open(currentPath(), std::ios::out | std::ios::app);
        if (!stream_.is_open()) {
            throw std::runtime_error("Cannot open cross-camera log file: " +
                                     currentPath().string());
        }
    }

    void rotateIfNeeded()
    {
        const auto position = stream_.tellp();
        if (position >= kMaxFileSize) {
            stream_.close();
            ++part_;
            openFile();
        }
    }

    void write(const char *level,
               const std::string &event,
               const std::string &message,
               bool console)
    {
        if (!enabled_) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        rotateIfNeeded();
        const std::string line = currentTimeForLine() + " [" + level + "] [" +
                                 event + "] " + message;
        stream_ << line << '\n';
        stream_.flush();
        if (console) {
            std::cerr << line << std::endl;
        }
    }
};

class DisjointSet {
  public:
    explicit DisjointSet(size_t count)
        : parent_(count), rank_(count, 0)
    {
        for (size_t i = 0; i < count; ++i) {
            parent_[i] = i;
        }
    }

    size_t find(size_t value)
    {
        if (parent_[value] != value) {
            parent_[value] = find(parent_[value]);
        }
        return parent_[value];
    }

    size_t unite(size_t first, size_t second)
    {
        first = find(first);
        second = find(second);
        if (first == second) {
            return first;
        }
        if (rank_[first] < rank_[second]) {
            std::swap(first, second);
        }
        parent_[second] = first;
        if (rank_[first] == rank_[second]) {
            ++rank_[first];
        }
        return first;
    }

  private:
    std::vector<size_t> parent_;
    std::vector<size_t> rank_;
};

} // namespace

class CrossCameraTrackerPrivate {
  public:
    explicit CrossCameraTrackerPrivate(const CrossCameraTrackerConfig &config)
        : config_(config),
          logger_(config.enable_log, config.log_directory)
    {
        validateAndInitialize();
    }

    void update(const std::vector<CrossCameraFrameInput> &batch,
                std::vector<CrossCameraTrackedObject> &trackedObjects)
    {
        validateBatch(batch);
        logger_.info("UPDATE_BEGIN", "cameras=" + std::to_string(batch.size()));

        std::map<CameraId, const CrossCameraFrameInput *> inputs;
        for (const auto &input : batch) {
            inputs.emplace(input.camera_id, &input);
        }

        std::vector<ObservationState> observations;
        std::map<CameraId, std::vector<size_t>> observationIndices;

        for (auto &[cameraId, channel] : channels_) {
            observationIndices.emplace(cameraId, std::vector<size_t>{});
            const auto &input = *inputs.at(cameraId);
            std::vector<TrackedObject> localTracks;
            channel.tracker->update(input.detections, input.frame, localTracks);
            removeStaleBindings(cameraId, *channel.tracker);

            for (auto &localTrack : localTracks) {
                const cv::Point2f footPoint(
                    localTrack.bbox.x + localTrack.bbox.width / 2.0f,
                    localTrack.bbox.y + static_cast<float>(localTrack.bbox.height));

                ObservationState observation;
                observation.object.camera_id = cameraId;
                observation.object.local_track = std::move(localTrack);
                observation.object.mapped_point =
                    channel.homography->project_point(footPoint);
                observation.key = {
                    cameraId, observation.object.local_track.track_id};

                const auto binding = local_to_target_.find(observation.key);
                if (binding != local_to_target_.end()) {
                    observation.existing_target_id = binding->second;
                }

                observationIndices[cameraId].push_back(observations.size());
                observations.push_back(std::move(observation));
            }

            logger_.info("LOCAL_TRACK_SUMMARY",
                         "camera=" + std::to_string(cameraId) +
                             " tracks=" + std::to_string(localTracks.size()));
        }

        DisjointSet groups(observations.size());
        std::vector<CandidateMatch> candidates =
            buildCandidates(observations, observationIndices);
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidateMatch &left, const CandidateMatch &right) {
                      return left.normalized_distance <
                             right.normalized_distance;
                  });

        for (const auto &candidate : candidates) {
            tryMerge(candidate, observations, groups);
        }

        assignTargetIds(observations, groups);

        trackedObjects.clear();
        trackedObjects.reserve(observations.size());
        for (auto &observation : observations) {
            trackedObjects.push_back(std::move(observation.object));
        }
        std::sort(trackedObjects.begin(), trackedObjects.end(),
                  [](const CrossCameraTrackedObject &left,
                     const CrossCameraTrackedObject &right) {
                      return std::tie(left.camera_id,
                                      left.local_track.track_id) <
                             std::tie(right.camera_id,
                                      right.local_track.track_id);
                  });

        logger_.info("UPDATE_END",
                     "objects=" + std::to_string(trackedObjects.size()) +
                         " candidates=" + std::to_string(candidates.size()));
    }

  private:
    struct ChannelState {
        std::unique_ptr<Tracker> tracker;
        std::unique_ptr<Homography> homography;
    };

    struct ObservationState {
        CrossCameraTrackedObject object;
        LocalTrackKey key;
        TargetId existing_target_id = 0;
    };

    struct CandidateMatch {
        size_t first = 0;
        size_t second = 0;
        float distance = 0.0f;
        float normalized_distance = 0.0f;
    };

    CrossCameraTrackerConfig config_;
    FileLogger logger_;
    std::map<CameraId, ChannelState> channels_;
    std::map<LocalTrackKey, TargetId> local_to_target_;
    std::map<TargetId, std::set<LocalTrackKey>> target_members_;
    TargetId next_target_id_ = 1;

    [[noreturn]] void inputError(const std::string &message)
    {
        logger_.error("INPUT_ERROR", message);
        throw std::invalid_argument(message);
    }

    void validateAndInitialize()
    {
        if (config_.channels.empty()) {
            inputError("CrossCameraTracker requires at least one camera");
        }

        for (const auto &channelConfig : config_.channels) {
            if (channelConfig.tracker_fps <= 0.0f) {
                inputError("tracker_fps must be positive for camera " +
                           std::to_string(channelConfig.camera_id));
            }
            if (channelConfig.calibration_points.size() < 4) {
                inputError("At least four calibration points are required for camera " +
                           std::to_string(channelConfig.camera_id));
            }
            if (channels_.count(channelConfig.camera_id) != 0) {
                inputError("Duplicate camera_id: " +
                           std::to_string(channelConfig.camera_id));
            }

            std::vector<PointPair> pointPairs;
            pointPairs.reserve(channelConfig.calibration_points.size());
            for (const auto &point : channelConfig.calibration_points) {
                pointPairs.emplace_back(point.image_point, point.map_point);
            }

            ChannelState channel;
            channel.tracker = std::make_unique<Tracker>(
                config_.tracker_config, channelConfig.tracker_fps);
            channel.homography = std::make_unique<Homography>();
            if (channel.homography->compute(pointPairs).empty()) {
                throw std::runtime_error(
                    "Failed to compute homography for camera " +
                    std::to_string(channelConfig.camera_id));
            }

            channels_.emplace(channelConfig.camera_id, std::move(channel));
            logger_.info("CHANNEL_INITIALIZED",
                         "camera=" + std::to_string(channelConfig.camera_id));
        }

        std::set<std::pair<CameraId, CameraId>> uniqueLinks;
        for (auto &link : config_.links) {
            if (link.camera_a_id == link.camera_b_id) {
                inputError("A camera cannot be linked to itself");
            }
            if (link.max_distance <= 0.0f) {
                inputError("Link max_distance must be positive");
            }
            if (channels_.count(link.camera_a_id) == 0 ||
                channels_.count(link.camera_b_id) == 0) {
                inputError("Link references an unknown camera");
            }
            const auto normalized = std::minmax(link.camera_a_id,
                                                link.camera_b_id);
            if (!uniqueLinks.emplace(normalized.first, normalized.second).second) {
                inputError("Duplicate camera link");
            }
            logger_.info("LINK_INITIALIZED",
                         "camera_a=" + std::to_string(link.camera_a_id) +
                             " camera_b=" + std::to_string(link.camera_b_id) +
                             " max_distance=" +
                             std::to_string(link.max_distance));
        }

        logger_.info("TRACKER_INITIALIZED",
                     "cameras=" + std::to_string(channels_.size()) +
                         " links=" + std::to_string(config_.links.size()));
    }

    void validateBatch(const std::vector<CrossCameraFrameInput> &batch)
    {
        if (batch.size() != channels_.size()) {
            inputError("Batch must contain every configured camera exactly once");
        }

        std::set<CameraId> seen;
        for (const auto &input : batch) {
            if (channels_.count(input.camera_id) == 0) {
                inputError("Batch contains unknown camera_id: " +
                           std::to_string(input.camera_id));
            }
            if (!seen.insert(input.camera_id).second) {
                inputError("Batch contains duplicate camera_id: " +
                           std::to_string(input.camera_id));
            }
            if (input.frame.empty()) {
                inputError("Batch contains an empty frame for camera " +
                           std::to_string(input.camera_id));
            }
        }
    }

    void removeStaleBindings(CameraId cameraId, Tracker &tracker)
    {
        std::vector<size_t> removedIds;
        tracker.get_removed_ids(removedIds);
        for (const size_t localId : removedIds) {
            const LocalTrackKey key{cameraId, localId};
            const auto binding = local_to_target_.find(key);
            if (binding == local_to_target_.end()) {
                continue;
            }

            const TargetId targetId = binding->second;
            local_to_target_.erase(binding);
            auto members = target_members_.find(targetId);
            if (members != target_members_.end()) {
                members->second.erase(key);
                if (members->second.empty()) {
                    target_members_.erase(members);
                    logger_.info("TARGET_REMOVED",
                                 "target_id=" + std::to_string(targetId));
                }
            }
            logger_.info("LOCAL_BINDING_REMOVED",
                         "camera=" + std::to_string(cameraId) +
                             " local_id=" + std::to_string(localId) +
                             " target_id=" + std::to_string(targetId));
        }
    }

    std::vector<CandidateMatch> buildCandidates(
        const std::vector<ObservationState> &observations,
        const std::map<CameraId, std::vector<size_t>> &indices)
    {
        std::vector<CandidateMatch> candidates;
        for (const auto &link : config_.links) {
            const auto &firstIndices = indices.at(link.camera_a_id);
            const auto &secondIndices = indices.at(link.camera_b_id);
            if (firstIndices.empty() || secondIndices.empty()) {
                continue;
            }

            const size_t columns = firstIndices.size();
            const size_t rows = secondIndices.size();
            distMatrix_t costs(columns * rows, 2.0f);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    const auto &first = observations[firstIndices[column]].object;
                    const auto &second = observations[secondIndices[row]].object;
                    if (first.local_track.class_id !=
                        second.local_track.class_id) {
                        continue;
                    }

                    const float distance =
                        static_cast<float>(cv::norm(first.mapped_point -
                                                    second.mapped_point));
                    logger_.debug(
                        "MATCH_CANDIDATE",
                        "camera_a=" + std::to_string(first.camera_id) +
                            " local_a=" +
                            std::to_string(first.local_track.track_id) +
                            " camera_b=" + std::to_string(second.camera_id) +
                            " local_b=" +
                            std::to_string(second.local_track.track_id) +
                            " distance=" + std::to_string(distance));
                    if (distance <= link.max_distance) {
                        costs[column + row * columns] =
                            distance / link.max_distance;
                    }
                }
            }

            assignments_t assignments(columns, -1);
            SPLAPJV solver({1.0f, 1});
            solver.Solve(costs, columns, rows, assignments, 2.0f);
            for (size_t column = 0; column < assignments.size(); ++column) {
                const int row = assignments[column];
                if (row < 0) {
                    continue;
                }
                const float normalized = costs[column +
                                               static_cast<size_t>(row) *
                                                   columns];
                if (normalized > 1.0f) {
                    continue;
                }
                candidates.push_back({
                    firstIndices[column],
                    secondIndices[static_cast<size_t>(row)],
                    normalized * link.max_distance,
                    normalized,
                });
            }
        }
        return candidates;
    }

    static std::set<CameraId> groupCameras(
        size_t root,
        const std::vector<ObservationState> &observations,
        DisjointSet &groups)
    {
        std::set<CameraId> cameras;
        for (size_t i = 0; i < observations.size(); ++i) {
            if (groups.find(i) == root) {
                cameras.insert(observations[i].key.camera_id);
            }
        }
        return cameras;
    }

    static std::set<TargetId> groupTargetIds(
        size_t root,
        const std::vector<ObservationState> &observations,
        DisjointSet &groups)
    {
        std::set<TargetId> targetIds;
        for (size_t i = 0; i < observations.size(); ++i) {
            if (groups.find(i) == root &&
                observations[i].existing_target_id != 0) {
                targetIds.insert(observations[i].existing_target_id);
            }
        }
        return targetIds;
    }

    void tryMerge(const CandidateMatch &candidate,
                  const std::vector<ObservationState> &observations,
                  DisjointSet &groups)
    {
        const size_t firstRoot = groups.find(candidate.first);
        const size_t secondRoot = groups.find(candidate.second);
        if (firstRoot == secondRoot) {
            return;
        }

        const auto firstCameras =
            groupCameras(firstRoot, observations, groups);
        const auto secondCameras =
            groupCameras(secondRoot, observations, groups);
        std::vector<CameraId> duplicateCameras;
        std::set_intersection(firstCameras.begin(), firstCameras.end(),
                              secondCameras.begin(), secondCameras.end(),
                              std::back_inserter(duplicateCameras));
        if (!duplicateCameras.empty()) {
            logger_.warning("MATCH_REJECTED_CONFLICT",
                            "reason=duplicate_camera distance=" +
                                std::to_string(candidate.distance));
            return;
        }

        auto targetIds = groupTargetIds(firstRoot, observations, groups);
        const auto secondTargetIds =
            groupTargetIds(secondRoot, observations, groups);
        targetIds.insert(secondTargetIds.begin(), secondTargetIds.end());
        if (targetIds.size() > 1) {
            logger_.warning("MATCH_REJECTED_CONFLICT",
                            "reason=different_target_ids distance=" +
                                std::to_string(candidate.distance));
            return;
        }

        groups.unite(firstRoot, secondRoot);
        const auto &first = observations[candidate.first];
        const auto &second = observations[candidate.second];
        logger_.info(
            "MATCH_ACCEPTED",
            "camera_a=" + std::to_string(first.key.camera_id) +
                " local_a=" + std::to_string(first.key.local_id) +
                " camera_b=" + std::to_string(second.key.camera_id) +
                " local_b=" + std::to_string(second.key.local_id) +
                " distance=" + std::to_string(candidate.distance));
    }

    void assignTargetIds(std::vector<ObservationState> &observations,
                         DisjointSet &groups)
    {
        std::map<size_t, std::vector<size_t>> membersByRoot;
        for (size_t i = 0; i < observations.size(); ++i) {
            membersByRoot[groups.find(i)].push_back(i);
        }

        for (const auto &[root, members] : membersByRoot) {
            (void)root;
            TargetId targetId = 0;
            for (const size_t index : members) {
                if (observations[index].existing_target_id != 0) {
                    targetId = observations[index].existing_target_id;
                    break;
                }
            }
            if (targetId == 0) {
                targetId = next_target_id_++;
                logger_.info("TARGET_CREATED",
                             "target_id=" + std::to_string(targetId));
            }

            for (const size_t index : members) {
                auto &observation = observations[index];
                observation.object.target_id = targetId;
                local_to_target_[observation.key] = targetId;
                target_members_[targetId].insert(observation.key);
                if (observation.existing_target_id == 0) {
                    logger_.info(
                        "TARGET_INHERITED",
                        "camera=" +
                            std::to_string(observation.key.camera_id) +
                            " local_id=" +
                            std::to_string(observation.key.local_id) +
                            " target_id=" + std::to_string(targetId));
                }
            }
        }
    }
};

CrossCameraTracker::CrossCameraTracker(
    const CrossCameraTrackerConfig &config)
    : m_pHandle(std::make_shared<CrossCameraTrackerPrivate>(config))
{
}

CrossCameraTracker::~CrossCameraTracker() = default;

void CrossCameraTracker::update(
    const std::vector<CrossCameraFrameInput> &batch,
    std::vector<CrossCameraTrackedObject> &tracked_objects)
{
    m_pHandle->update(batch, tracked_objects);
}

} // namespace JHDeepCore
