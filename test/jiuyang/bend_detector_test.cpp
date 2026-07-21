#include "billet_bend_detector.h"

#include <opencv2/imgproc.hpp>

#include <filesystem>
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
    config.debugDir = "result/bend_debug/merged";
    std::error_code ec;
    std::filesystem::create_directories(config.debugDir, ec);

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

    // Disconnected billets must each use their own foreground pixels for PCA.
    cv::Mat separatedMask = cv::Mat::zeros(480, 520, CV_8U);
    cv::line(separatedMask, cv::Point(70, 440), cv::Point(150, 45),
             cv::Scalar(255), 24, cv::LINE_8);
    cv::line(separatedMask, cv::Point(260, 440), cv::Point(260, 45),
             cv::Scalar(255), 24, cv::LINE_8);
    cv::line(separatedMask, cv::Point(450, 440), cv::Point(370, 45),
             cv::Scalar(255), 24, cv::LINE_8);

    JHDeepCore::BilletBendConfig separatedConfig = config;
    separatedConfig.debugDir = "result/bend_debug/separated";
    std::filesystem::create_directories(separatedConfig.debugDir, ec);
    JHDeepCore::BilletBendDetector separatedDetector(separatedConfig);
    const auto separatedResults = separatedDetector.detect(separatedMask);
    int separatedValidCount = 0;
    for (const auto &result : separatedResults) {
        if (result.valid) ++separatedValidCount;
    }
    if (separatedResults.size() != 3 || separatedValidCount != 3) {
        std::cerr << "Disconnected-component PCA failed: results="
                  << separatedResults.size() << ", valid=" << separatedValidCount << std::endl;
        return 1;
    }
    return 0;
}
