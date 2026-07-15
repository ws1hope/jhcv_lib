#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace JHDeepCore {

struct BilletBendConfig {
    int expectedBilletCount = 0;
    int minConnectedArea = 100;
    int minTrackPoints = 20;
    int maxMissingSlices = 2;

    float sliceStep = 4.0f;
    float sliceHalfThickness = 2.5f;
    float transverseResolution = 1.0f;
    float minRadius = 1.5f;
    float minPeakSpacing = 8.0f;
    float maxMatchDistance = 12.0f;
    float endTrimRatio = 0.05f;
    float minimumPeakCountRatio = 0.70f;
    float radiusIncreaseRatio = 1.60f;

    // No calibration is required: deviations are divided by local billet width.
    float bend95Threshold = 0.05f;
    float bendRmsThreshold = 0.025f;

    // The positive scan direction goes from the near end to the far end.
    // For the usual camera layout this is upwards in the image.
    cv::Point2f preferredDirection = cv::Point2f(0.0f, -1.0f);
};

struct BilletBendResult {
    int id = -1;
    bool valid = false;
    bool bent = false;
    bool stoppedAtMerge = false;

    float bend95 = 0.0f;
    float bendRms = 0.0f;
    float bendMax = 0.0f;
    float medianWidth = 0.0f;
    float validLengthRatio = 0.0f;

    cv::Vec4f fittedLine{};
    std::vector<cv::Point2f> centerPoints;
};

class BilletBendDetector {
  public:
    explicit BilletBendDetector(BilletBendConfig config = {});

    /// Detect bending directly from a combined binary mask. Foreground may contain
    /// several billets and may be connected at the far end.
    std::vector<BilletBendResult> detect(const cv::Mat &mask) const;

    /// Draw center samples, fitted reference lines and metrics.
    static void draw(cv::Mat &image, const std::vector<BilletBendResult> &results);

  private:
    BilletBendConfig config_;
};

} // namespace JHDeepCore
