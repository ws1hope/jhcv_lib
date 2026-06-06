#include "JHDeepCore.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace JHDeepCore;

namespace {

std::vector<CalibrationPointPair> identityCalibration()
{
    return {
        {{0.0f, 0.0f}, {0.0f, 0.0f}},
        {{1000.0f, 0.0f}, {1000.0f, 0.0f}},
        {{1000.0f, 1000.0f}, {1000.0f, 1000.0f}},
        {{0.0f, 1000.0f}, {0.0f, 1000.0f}},
    };
}

Detection makeDetection(int x, int y, int classId = 0)
{
    Detection detection;
    detection.bbox = cv::Rect(x, y, 40, 80);
    detection.confidence = 0.9f;
    detection.class_id = classId;
    detection.class_name = std::to_string(classId);
    return detection;
}

CrossCameraTrackerConfig makeConfig()
{
    CrossCameraTrackerConfig config;
    config.tracker_config.tracker_type = TrackerType::ByteTrack;
    config.enable_log = false;
    config.channels = {
        {1, 25.0f, identityCalibration()},
        {2, 25.0f, identityCalibration()},
        {3, 25.0f, identityCalibration()},
    };
    config.links = {
        {1, 2, 200.0f},
        {2, 3, 200.0f},
    };
    return config;
}

CrossCameraFrameInput makeInput(CameraId cameraId,
                                std::vector<Detection> detections)
{
    CrossCameraFrameInput input;
    input.camera_id = cameraId;
    input.frame = cv::Mat::zeros(1000, 1000, CV_8UC3);
    input.detections = std::move(detections);
    return input;
}

const CrossCameraTrackedObject& findCamera(
    const std::vector<CrossCameraTrackedObject>& objects,
    CameraId cameraId)
{
    for (const auto& object : objects) {
        if (object.camera_id == cameraId) {
            return object;
        }
    }
    assert(false && "camera result not found");
    return objects.front();
}

const CrossCameraTrackedObject& findLocal(
    const std::vector<CrossCameraTrackedObject>& objects,
    CameraId cameraId,
    size_t localId)
{
    for (const auto& object : objects) {
        if (object.camera_id == cameraId &&
            object.local_track.track_id == localId) {
            return object;
        }
    }
    assert(false && "local track result not found");
    return objects.front();
}

void testAdjacentTargetsShareTargetId()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;

    tracker.update({
        makeInput(1, {makeDetection(100, 100)}),
        makeInput(2, {makeDetection(120, 110)}),
        makeInput(3, {}),
    }, objects);

    assert(objects.size() == 2);
    const auto& cam1 = findCamera(objects, 1);
    const auto& cam2 = findCamera(objects, 2);
    assert(cam1.target_id == cam2.target_id);
    assert(cam1.local_track.track_id == 1);
    assert(cam2.local_track.track_id == 1);
    const cv::Point2f expectedCenter(
        cam1.local_track.bbox.x + cam1.local_track.bbox.width / 2.0f,
        cam1.local_track.bbox.y + cam1.local_track.bbox.height / 2.0f);
    assert(cv::norm(cam1.mapped_point - expectedCenter) < 0.01f);
    assert(!cam1.local_track.trajectory.empty());
    assert(cv::norm(
               cv::Point2f(cam1.local_track.trajectory.back()) -
               expectedCenter) < 1.0f);
}

void testNonAdjacentTargetsStaySeparate()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;

    tracker.update({
        makeInput(1, {makeDetection(100, 100)}),
        makeInput(2, {}),
        makeInput(3, {makeDetection(100, 100)}),
    }, objects);

    assert(objects.size() == 2);
    assert(findCamera(objects, 1).target_id != findCamera(objects, 3).target_id);
}

void testClassAndDistanceAreHardConstraints()
{
    CrossCameraTracker classTracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;
    classTracker.update({
        makeInput(1, {makeDetection(100, 100, 0)}),
        makeInput(2, {makeDetection(110, 100, 1)}),
        makeInput(3, {}),
    }, objects);
    assert(findCamera(objects, 1).target_id != findCamera(objects, 2).target_id);

    CrossCameraTracker distanceTracker(makeConfig());
    distanceTracker.update({
        makeInput(1, {makeDetection(100, 100)}),
        makeInput(2, {makeDetection(500, 500)}),
        makeInput(3, {}),
    }, objects);
    assert(findCamera(objects, 1).target_id != findCamera(objects, 2).target_id);
}

void testAdjacentChainSharesOneTargetId()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;

    tracker.update({
        makeInput(1, {makeDetection(100, 100)}),
        makeInput(2, {makeDetection(120, 100)}),
        makeInput(3, {makeDetection(140, 100)}),
    }, objects);

    assert(objects.size() == 3);
    std::set<TargetId> targetIds;
    for (const auto& object : objects) {
        targetIds.insert(object.target_id);
    }
    assert(targetIds.size() == 1);
}

void testLapjvKeepsMultipleTargetsOneToOne()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;

    tracker.update({
        makeInput(1, {
            makeDetection(100, 100),
            makeDetection(500, 100),
        }),
        makeInput(2, {
            makeDetection(510, 100),
            makeDetection(110, 100),
        }),
        makeInput(3, {}),
    }, objects);

    assert(objects.size() == 4);
    assert(findLocal(objects, 1, 1).target_id ==
           findLocal(objects, 2, 2).target_id);
    assert(findLocal(objects, 1, 2).target_id ==
           findLocal(objects, 2, 1).target_id);
    assert(findLocal(objects, 1, 1).target_id !=
           findLocal(objects, 1, 2).target_id);
}

void testTargetIdPersistsAcrossUpdates()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;

    tracker.update({
        makeInput(1, {makeDetection(100, 100)}),
        makeInput(2, {makeDetection(120, 100)}),
        makeInput(3, {}),
    }, objects);
    const TargetId firstTargetId = findCamera(objects, 1).target_id;

    tracker.update({
        makeInput(1, {makeDetection(102, 100)}),
        makeInput(2, {makeDetection(122, 100)}),
        makeInput(3, {}),
    }, objects);
    assert(findCamera(objects, 1).target_id == firstTargetId);
    assert(findCamera(objects, 2).target_id == firstTargetId);
}

void testInvalidBatchThrows()
{
    CrossCameraTracker tracker(makeConfig());
    std::vector<CrossCameraTrackedObject> objects;
    bool threw = false;
    try {
        tracker.update({
            makeInput(1, {}),
            makeInput(2, {}),
        }, objects);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testLoggingCanBeDisabled()
{
    const auto logDirectory =
        std::filesystem::temp_directory_path() /
        "jhcv_cross_camera_disabled_logs";
    std::filesystem::remove_all(logDirectory);

    auto config = makeConfig();
    config.enable_log = false;
    config.log_directory = logDirectory.string();
    CrossCameraTracker tracker(config);
    assert(!std::filesystem::exists(logDirectory));
}

void testLoggingCreatesAFileWhenEnabled()
{
    const auto logDirectory =
        std::filesystem::temp_directory_path() /
        "jhcv_cross_camera_enabled_logs";
    std::filesystem::remove_all(logDirectory);

    auto config = makeConfig();
    config.enable_log = true;
    config.log_directory = logDirectory.string();
    {
        CrossCameraTracker tracker(config);
    }

    assert(std::filesystem::exists(logDirectory));
    assert(std::filesystem::directory_iterator(logDirectory) !=
           std::filesystem::directory_iterator());
    std::filesystem::remove_all(logDirectory);
}

} // namespace

int main()
{
    testAdjacentTargetsShareTargetId();
    testNonAdjacentTargetsStaySeparate();
    testClassAndDistanceAreHardConstraints();
    testAdjacentChainSharesOneTargetId();
    testLapjvKeepsMultipleTargetsOneToOne();
    testTargetIdPersistsAcrossUpdates();
    testInvalidBatchThrows();
    testLoggingCanBeDisabled();
    testLoggingCreatesAFileWhenEnabled();
    std::cout << "cross_camera_tracker_test passed" << std::endl;
    return 0;
}
