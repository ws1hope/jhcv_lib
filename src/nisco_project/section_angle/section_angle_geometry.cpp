#include "section_angle_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace JHDeepCore {
namespace section_angle {

namespace {

cv::Point2f computeContourCenter(const std::vector<cv::Point> &contour)
{
    cv::Moments moments = cv::moments(contour);
    if (std::abs(moments.m00) < 1e-6) {
        return cv::Point2f(0.f, 0.f);
    }
    return cv::Point2f(
        static_cast<float>(moments.m10 / moments.m00),
        static_cast<float>(moments.m01 / moments.m00));
}

float pointToLineDistance(const cv::Point2f &p, const Line2D &line)
{
    cv::Point2f vec(p.x - line.point.x, p.y - line.point.y);
    cv::Point2f normal(-line.direction.y, line.direction.x);
    return std::abs(vec.x * normal.x + vec.y * normal.y);
}

float pointToSegmentDistance(const cv::Point2f &p, const cv::Point2f &a, const cv::Point2f &b)
{
    cv::Point2f ab = b - a;
    float len2 = ab.x * ab.x + ab.y * ab.y;
    if (len2 < 1e-6f) {
        return std::hypot(p.x - a.x, p.y - a.y);
    }
    float t = std::max(0.f, std::min(1.f, ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2));
    cv::Point2f proj(a.x + t * ab.x, a.y + t * ab.y);
    return std::hypot(p.x - proj.x, p.y - proj.y);
}

void orderCornersClockwise(cv::Point2f corners[4])
{
    cv::Point2f center(0.f, 0.f);
    for (int i = 0; i < 4; ++i) {
        center += corners[i];
    }
    center *= 0.25f;

    std::array<std::pair<float, int>, 4> indexed;
    for (int i = 0; i < 4; ++i) {
        indexed[i] = {
            std::atan2(corners[i].y - center.y, corners[i].x - center.x),
            i
        };
    }
    std::sort(indexed.begin(), indexed.end());

    cv::Point2f ordered[4];
    for (int i = 0; i < 4; ++i) {
        ordered[i] = corners[indexed[i].second];
    }
    for (int i = 0; i < 4; ++i) {
        corners[i] = ordered[i];
    }
}

double computeOutlierThreshold(const std::vector<cv::Point> &contour)
{
    cv::RotatedRect rect = cv::minAreaRect(contour);
    double shorter = std::min(rect.size.width, rect.size.height);
    return std::max(2.0, shorter * 0.05);
}

bool fitLineRobust(const std::vector<cv::Point2f> &points,
                   Line2D &line,
                   std::vector<cv::Point2f> &inliers,
                   double outlier_thresh,
                   int max_iterations)
{
    if (points.size() < 2) {
        return false;
    }

    inliers = points;
    for (int iter = 0; iter < max_iterations; ++iter) {
        cv::Vec4f params;
        cv::fitLine(inliers, params, cv::DIST_HUBER, 0, 0.01, 0.01);

        cv::Point2f dir(params[0], params[1]);
        float len = std::hypot(dir.x, dir.y);
        if (len < 1e-6f) {
            return false;
        }
        dir.x /= len;
        dir.y /= len;
        line.direction = dir;

        std::vector<cv::Point2f> next_inliers;
        next_inliers.reserve(inliers.size());
        for (const auto &point : inliers) {
            if (pointToLineDistance(point, line) <= static_cast<float>(outlier_thresh)) {
                next_inliers.push_back(point);
            }
        }

        if (next_inliers.size() < 2) {
            break;
        }
        if (next_inliers.size() == inliers.size()) {
            inliers = next_inliers;
            break;
        }
        inliers = std::move(next_inliers);
    }

    cv::Point2f centroid(0.f, 0.f);
    for (const auto &point : inliers) {
        centroid += point;
    }
    centroid.x /= static_cast<float>(inliers.size());
    centroid.y /= static_cast<float>(inliers.size());
    line.point = centroid;
    return inliers.size() >= 2;
}

bool intersectLines(const Line2D &line1, const Line2D &line2, cv::Point2f &intersection)
{
    float cross = line1.direction.x * line2.direction.y - line1.direction.y * line2.direction.x;
    if (std::abs(cross) < 1e-5f) {
        return false;
    }

    cv::Point2f diff(line2.point.x - line1.point.x, line2.point.y - line1.point.y);
    float t = (diff.x * line2.direction.y - diff.y * line2.direction.x) / cross;
    intersection.x = line1.point.x + line1.direction.x * t;
    intersection.y = line1.point.y + line1.direction.y * t;
    return true;
}

std::array<std::vector<cv::Point2f>, 4> assignPointsToEdges(
    const std::vector<cv::Point> &contour,
    const cv::Point2f corners[4])
{
    std::array<std::vector<cv::Point2f>, 4> edge_points;
    for (const auto &pt : contour) {
        cv::Point2f point(static_cast<float>(pt.x), static_cast<float>(pt.y));
        int best_edge = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int edge_idx = 0; edge_idx < 4; ++edge_idx) {
            int next = (edge_idx + 1) % 4;
            float dist = pointToSegmentDistance(point, corners[edge_idx], corners[next]);
            if (dist < best_dist) {
                best_dist = dist;
                best_edge = edge_idx;
            }
        }
        edge_points[best_edge].push_back(point);
    }
    return edge_points;
}

bool fitQuadrilateralByLines(const std::vector<cv::Point> &contour,
                             cv::Point2f corners[4],
                             Line2D fitted_lines[4],
                             int edge_point_counts[4],
                             const SectionAngleConfig &config,
                             std::string &error_msg)
{
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f init_corners[4];
    rect.points(init_corners);
    orderCornersClockwise(init_corners);

    const double outlier_thresh = computeOutlierThreshold(contour);
    auto edge_points = assignPointsToEdges(contour, init_corners);

    for (int edge_idx = 0; edge_idx < 4; ++edge_idx) {
        edge_point_counts[edge_idx] = static_cast<int>(edge_points[edge_idx].size());
        if (edge_point_counts[edge_idx] < config.min_edge_inlier_points) {
            error_msg = "Edge " + std::to_string(edge_idx + 1) + " has too few points: "
                        + std::to_string(edge_points[edge_idx].size());
            return false;
        }

        std::vector<cv::Point2f> inliers;
        if (!fitLineRobust(edge_points[edge_idx], fitted_lines[edge_idx], inliers,
                           outlier_thresh, config.line_fit_max_iterations)) {
            error_msg = "Failed to fit line for edge " + std::to_string(edge_idx + 1);
            return false;
        }

        edge_point_counts[edge_idx] = static_cast<int>(inliers.size());
        if (edge_point_counts[edge_idx] < config.min_edge_inlier_points) {
            error_msg = "Edge " + std::to_string(edge_idx + 1)
                        + " has too few inliers after fitting: "
                        + std::to_string(edge_point_counts[edge_idx]);
            return false;
        }
    }

    for (int corner_idx = 0; corner_idx < 4; ++corner_idx) {
        int prev_edge = (corner_idx + 3) % 4;
        if (!intersectLines(fitted_lines[prev_edge], fitted_lines[corner_idx], corners[corner_idx])) {
            error_msg = "Failed to intersect fitted lines at corner "
                        + std::to_string(corner_idx + 1);
            return false;
        }
    }

    return true;
}

float computeInteriorAngle(const cv::Point2f &prev,
                           const cv::Point2f &curr,
                           const cv::Point2f &next)
{
    cv::Point2f v1 = prev - curr;
    cv::Point2f v2 = next - curr;
    double len1 = std::hypot(v1.x, v1.y);
    double len2 = std::hypot(v2.x, v2.y);
    if (len1 < 1e-6 || len2 < 1e-6) {
        return 0.f;
    }

    double dot = v1.x * v2.x + v1.y * v2.y;
    double cos_angle = dot / (len1 * len2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return static_cast<float>(std::acos(cos_angle) * 180.0 / CV_PI);
}

void fillAnglesFromCorners(const cv::Point2f corners[4], float angles[4])
{
    for (int i = 0; i < 4; ++i) {
        int prev = (i + 3) % 4;
        int next = (i + 1) % 4;
        angles[i] = computeInteriorAngle(corners[prev], corners[i], corners[next]);
    }
}

void fillFromMinAreaRect(const std::vector<cv::Point> &contour, ContourAngleDetail &out)
{
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f pts[4];
    rect.points(pts);
    for (int i = 0; i < 4; ++i) {
        out.corners[i] = pts[i];
    }
    fillAnglesFromCorners(out.corners, out.angles);
    out.used_fallback = true;
}

bool checkRightAngles(const float angles[4],
                      float tolerance_deg,
                      std::vector<std::string> &alerts)
{
    bool all_right = true;
    for (int i = 0; i < 4; ++i) {
        float diff = std::abs(angles[i] - 90.f);
        if (diff > tolerance_deg) {
            all_right = false;
            std::ostringstream oss;
            oss << "Right-angle violation: corner " << (i + 1)
                << " = " << std::fixed << std::setprecision(1) << angles[i]
                << " deg (tolerance +/-" << tolerance_deg << " deg)";
            alerts.push_back(oss.str());
        }
    }
    return all_right;
}

} // namespace

cv::Mat extractClassMask(const SegmentationResult &result, int class_id)
{
    cv::Mat mask = (result.segmentation_mask == class_id);
    cv::Mat binary;
    mask.convertTo(binary, CV_8UC1, 255);
    return binary;
}

std::vector<std::vector<cv::Point>> findClassContours(const cv::Mat &mask, double min_area)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> valid_contours;
    valid_contours.reserve(contours.size());
    for (auto &contour : contours) {
        if (cv::contourArea(contour) >= min_area) {
            valid_contours.push_back(contour);
        }
    }

    std::sort(valid_contours.begin(), valid_contours.end(),
              [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });
    return valid_contours;
}

ContourAngleDetail analyzeContour(const std::vector<cv::Point> &contour,
                                int instance_id,
                                const SectionAngleConfig &config)
{
    ContourAngleDetail detail;
    detail.instance_id = instance_id;
    detail.contour = contour;
    detail.center = computeContourCenter(contour);

    if (contour.empty()) {
        detail.alerts.push_back("Empty contour");
        return detail;
    }

    std::string fit_error;
    if (fitQuadrilateralByLines(contour, detail.corners, detail.fitted_lines,
                                detail.edge_inlier_counts, config, fit_error)) {
        detail.is_valid_quad = true;
        detail.vertex_count = 4;
        fillAnglesFromCorners(detail.corners, detail.angles);
    } else {
        detail.alerts.push_back("Line fitting failed: " + fit_error);
        fillFromMinAreaRect(contour, detail);
        std::ostringstream fb;
        fb << "Fallback minAreaRect angles (approximate): "
           << std::fixed << std::setprecision(1)
           << detail.angles[0] << ", " << detail.angles[1] << ", "
           << detail.angles[2] << ", " << detail.angles[3];
        detail.alerts.push_back(fb.str());
    }

    detail.has_right_angles = checkRightAngles(detail.angles, config.angle_tolerance_deg,
                                               detail.alerts);
    return detail;
}

std::vector<ContourAngleDetail> analyzeMask(const cv::Mat &mask,
                                           const cv::Size &image_shape,
                                           const SectionAngleConfig &config)
{
    (void)image_shape;
    std::vector<ContourAngleDetail> details;
    const double min_area = std::max(
        100.0,
        static_cast<double>(mask.cols * mask.rows) * config.min_area_ratio);

    std::vector<std::vector<cv::Point>> contours = findClassContours(mask, min_area);
    details.reserve(contours.size());

    for (size_t i = 0; i < contours.size(); ++i) {
        details.push_back(analyzeContour(contours[i], static_cast<int>(i) + 1, config));
    }

    return details;
}

SectionAngleItem toPublicItem(const ContourAngleDetail &detail)
{
    SectionAngleItem item;
    item.instance_id = detail.instance_id;
    for (int i = 0; i < 4; ++i) {
        item.corners[i] = detail.corners[i];
        item.angles[i] = detail.angles[i];
    }
    item.has_alert = !(detail.is_valid_quad && detail.has_right_angles);
    return item;
}

} // namespace section_angle
} // namespace JHDeepCore
