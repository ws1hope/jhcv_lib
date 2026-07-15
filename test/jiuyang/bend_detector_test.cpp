#include "billet_bend_detector.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

int main() {
    cv::Mat mask = cv::Mat::zeros(480, 420, CV_8U);

    // Two straight billets and one deliberately curved billet.
    cv::line(mask, cv::Point(90, 440), cv::Point(120, 45), cv::Scalar(255), 24, cv::LINE_AA);
    cv::line(mask, cv::Point(205, 440), cv::Point(210, 45), cv::Scalar(255), 24, cv::LINE_AA);
    std::vector<cv::Point> curved = {
        {330, 440}, {326, 390}, {318, 340}, {306, 290}, {292, 240},
        {282, 190}, {278, 140}, {282, 90}, {292, 45}};
    cv::polylines(mask, curved, false, cv::Scalar(255), 24, cv::LINE_AA);

    // Simulate the far-end segmentation merge.
    cv::line(mask, cv::Point(115, 50), cv::Point(295, 50), cv::Scalar(255), 12, cv::LINE_AA);

    JHDeepCore::BilletBendConfig config;
    config.expectedBilletCount = 3;
    config.minTrackPoints = 25;
    config.bend95Threshold = 0.035f;
    config.bendRmsThreshold = 0.018f;

    JHDeepCore::BilletBendDetector detector(config);
    const auto results = detector.detect(mask);

    int validCount = 0;
    int bentCount = 0;
    for (const auto &result : results) {
        if (!result.valid) continue;
        ++validCount;
        if (result.bent) ++bentCount;
        std::cout << "id=" << result.id << " bend95=" << result.bend95
                  << " bendRms=" << result.bendRms << " bent=" << result.bent << '\n';
    }

    if (validCount < 3 || bentCount < 1) {
        std::cerr << "Synthetic bend detection failed: valid=" << validCount
                  << ", bent=" << bentCount << std::endl;
        return 1;
    }
    return 0;
}
