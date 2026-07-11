#pragma once

#include "JHDeepCore.h"
#include <array>
#include <string>
#include <vector>

namespace JHDeepCore {
namespace section_angle {

struct Line2D {
    cv::Point2f point;
    cv::Point2f direction;
};

struct SectionAngleConfig {
    int target_class_id = 1;
    float angle_tolerance_deg = 8.0f;
    double min_area_ratio = 0.001;
    int min_edge_inlier_points = 8;
    int line_fit_max_iterations = 5;
};

struct ContourAngleDetail {
    int instance_id = 0;
    int vertex_count = 4;
    std::vector<cv::Point> contour;
    cv::Point2f corners[4];
    float angles[4] = {0.f, 0.f, 0.f, 0.f};
    cv::Point2f center;
    Line2D fitted_lines[4];
    int edge_inlier_counts[4] = {0, 0, 0, 0};
    bool is_valid_quad = false;
    bool has_right_angles = false;
    bool used_fallback = false;
    std::vector<std::string> alerts;
};

cv::Mat extractClassMask(const SegmentationResult &result, int class_id);
std::vector<std::vector<cv::Point>> findClassContours(const cv::Mat &mask, double min_area);
ContourAngleDetail analyzeContour(const std::vector<cv::Point> &contour,
                                  int instance_id,
                                  const SectionAngleConfig &config);
std::vector<ContourAngleDetail> analyzeMask(const cv::Mat &mask,
                                            const cv::Size &image_shape,
                                            const SectionAngleConfig &config);
SectionAngleItem toPublicItem(const ContourAngleDetail &detail);

} // namespace section_angle
} // namespace JHDeepCore
