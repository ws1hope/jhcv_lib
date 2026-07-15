#include "billet_bend_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace JHDeepCore {
namespace {

struct LocalPixel {
    float s;
    float t;
    cv::Point2f point;
    float radius;
};

struct Peak {
    float s;
    float t;
    cv::Point2f point;
    float radius;
};

struct Track {
    int id = -1;
    int missing = 0;
    bool active = true;
    bool stoppedAtMerge = false;
    std::vector<Peak> points;
};

cv::Mat prepareMask(const cv::Mat &input, int minArea) {
    if (input.empty()) {
        return {};
    }
    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input.clone();
    } else if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::invalid_argument("Billet mask must have 1, 3 or 4 channels");
    }
    if (gray.depth() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY);

    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(gray, labels, stats, centroids, 8, CV_32S);
    cv::Mat clean = cv::Mat::zeros(gray.size(), CV_8U);
    for (int label = 1; label < count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= minArea) {
            clean.setTo(255, labels == label);
        }
    }
    return clean;
}

float median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const float upper = values[mid];
    if (values.size() % 2 != 0) {
        return upper;
    }
    const float lower = *std::max_element(values.begin(), values.begin() + mid);
    return 0.5f * (lower + upper);
}

float pointLineDistance(const cv::Point2f &point, const cv::Vec4f &line) {
    const float norm = std::hypot(line[0], line[1]);
    if (norm < 1e-6f) {
        return 0.0f;
    }
    return std::abs((point.x - line[2]) * line[1] - (point.y - line[3]) * line[0]) / norm;
}

std::vector<float> smooth(const std::vector<float> &profile) {
    if (profile.size() < 5) {
        return profile;
    }
    std::vector<float> output(profile.size(), 0.0f);
    for (size_t i = 2; i + 2 < profile.size(); ++i) {
        output[i] = (profile[i - 2] + 2.0f * profile[i - 1] + 3.0f * profile[i] +
                     2.0f * profile[i + 1] + profile[i + 2]) /
                    9.0f;
    }
    return output;
}

} // namespace

BilletBendDetector::BilletBendDetector(BilletBendConfig config) : config_(std::move(config)) {
    if (config_.sliceStep <= 0.0f || config_.transverseResolution <= 0.0f) {
        throw std::invalid_argument("Billet bend detector sampling intervals must be positive");
    }
}

std::vector<BilletBendResult> BilletBendDetector::detect(const cv::Mat &inputMask) const {
    const cv::Mat mask = prepareMask(inputMask, config_.minConnectedArea);
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        return {};
    }

    cv::Mat distanceMap;
    cv::distanceTransform(mask, distanceMap, cv::DIST_L2, cv::DIST_MASK_5);
    std::vector<cv::Point> foreground;
    cv::findNonZero(mask, foreground);
    if (foreground.size() < 10) {
        return {};
    }

    cv::Mat data(static_cast<int>(foreground.size()), 2, CV_64F);
    for (int i = 0; i < static_cast<int>(foreground.size()); ++i) {
        data.at<double>(i, 0) = foreground[i].x;
        data.at<double>(i, 1) = foreground[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    const cv::Point2f origin(static_cast<float>(pca.mean.at<double>(0, 0)),
                             static_cast<float>(pca.mean.at<double>(0, 1)));

    cv::Point2f axes[2] = {
        {static_cast<float>(pca.eigenvectors.at<double>(0, 0)),
         static_cast<float>(pca.eigenvectors.at<double>(0, 1))},
        {static_cast<float>(pca.eigenvectors.at<double>(1, 0)),
         static_cast<float>(pca.eigenvectors.at<double>(1, 1))}};
    cv::Point2f preferred = config_.preferredDirection;
    const float preferredNorm = std::hypot(preferred.x, preferred.y);
    preferred = preferredNorm > 1e-6f ? preferred * (1.0f / preferredNorm) : cv::Point2f(0, -1);
    int longIndex = std::abs(axes[1].dot(preferred)) > std::abs(axes[0].dot(preferred)) ? 1 : 0;
    cv::Point2f lengthAxis = axes[longIndex];
    if (lengthAxis.dot(preferred) < 0.0f) {
        lengthAxis = -lengthAxis;
    }
    const cv::Point2f widthAxis(-lengthAxis.y, lengthAxis.x);

    float minS = std::numeric_limits<float>::max(), maxS = std::numeric_limits<float>::lowest();
    float minT = std::numeric_limits<float>::max(), maxT = std::numeric_limits<float>::lowest();
    std::vector<LocalPixel> pixels;
    pixels.reserve(foreground.size());
    for (const cv::Point &p : foreground) {
        const cv::Point2f point(static_cast<float>(p.x), static_cast<float>(p.y));
        const cv::Point2f delta = point - origin;
        const float s = delta.dot(lengthAxis);
        const float t = delta.dot(widthAxis);
        pixels.push_back({s, t, point, distanceMap.at<float>(p.y, p.x)});
        minS = std::min(minS, s);
        maxS = std::max(maxS, s);
        minT = std::min(minT, t);
        maxT = std::max(maxT, t);
    }

    auto extractPeaks = [&](float sliceS) {
        const int count = static_cast<int>(std::ceil((maxT - minT) /
                                                     config_.transverseResolution)) + 1;
        std::vector<float> profile(count, 0.0f);
        std::vector<cv::Point2f> points(count, cv::Point2f(-1, -1));
        for (const LocalPixel &pixel : pixels) {
            if (std::abs(pixel.s - sliceS) > config_.sliceHalfThickness) {
                continue;
            }
            const int index = cvRound((pixel.t - minT) / config_.transverseResolution);
            if (index >= 0 && index < count && pixel.radius > profile[index]) {
                profile[index] = pixel.radius;
                points[index] = pixel.point;
            }
        }
        const std::vector<float> filtered = smooth(profile);
        std::vector<int> candidates;
        for (int i = 2; i + 2 < count; ++i) {
            if (filtered[i] >= config_.minRadius && filtered[i] >= filtered[i - 1] &&
                filtered[i] >= filtered[i + 1] && filtered[i] >= filtered[i - 2] &&
                filtered[i] >= filtered[i + 2]) {
                candidates.push_back(i);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [&](int a, int b) { return filtered[a] > filtered[b]; });
        std::vector<int> selected;
        for (int candidate : candidates) {
            const bool close = std::any_of(selected.begin(), selected.end(), [&](int chosen) {
                return std::abs(candidate - chosen) * config_.transverseResolution <
                       config_.minPeakSpacing;
            });
            if (!close && points[candidate].x >= 0.0f) {
                selected.push_back(candidate);
            }
        }
        std::vector<Peak> peaks;
        for (int index : selected) {
            peaks.push_back({sliceS, minT + index * config_.transverseResolution,
                             points[index], profile[index]});
        }
        std::sort(peaks.begin(), peaks.end(), [](const Peak &a, const Peak &b) {
            return a.t < b.t;
        });
        return peaks;
    };

    const float totalLength = maxS - minS;
    const float startS = minS + config_.endTrimRatio * totalLength;
    const float endS = maxS - config_.endTrimRatio * totalLength;
    std::vector<std::pair<float, std::vector<Peak>>> slices;
    for (float s = startS; s <= endS; s += config_.sliceStep) {
        slices.emplace_back(s, extractPeaks(s));
    }
    if (slices.empty()) {
        return {};
    }

    // Initialize at the best separated near-end slice, not at the irregular end face.
    size_t init = 0;
    size_t bestCount = 0;
    const size_t searchEnd = std::max<size_t>(1, slices.size() / 3);
    for (size_t i = 0; i < searchEnd; ++i) {
        if (slices[i].second.size() > bestCount) {
            bestCount = slices[i].second.size();
            init = i;
        }
        if (config_.expectedBilletCount > 0 &&
            slices[i].second.size() == static_cast<size_t>(config_.expectedBilletCount)) {
            init = i;
            bestCount = slices[i].second.size();
            break;
        }
    }
    if (bestCount == 0) {
        return {};
    }
    const int referenceCount = config_.expectedBilletCount > 0
                                   ? config_.expectedBilletCount
                                   : static_cast<int>(bestCount);
    const int minimumCount = std::max(1, cvCeil(referenceCount * config_.minimumPeakCountRatio));
    std::vector<Track> tracks;
    for (int i = 0; i < static_cast<int>(slices[init].second.size()); ++i) {
        Track track;
        track.id = i;
        track.points.push_back(slices[init].second[i]);
        tracks.push_back(std::move(track));
    }

    for (size_t sliceIndex = init + 1; sliceIndex < slices.size(); ++sliceIndex) {
        const std::vector<Peak> &peaks = slices[sliceIndex].second;
        if (static_cast<int>(peaks.size()) < minimumCount) {
            for (Track &track : tracks) {
                if (track.active) track.stoppedAtMerge = true;
            }
            break;
        }
        std::vector<bool> used(peaks.size(), false);
        for (Track &track : tracks) {
            if (!track.active) continue;
            const Peak &last = track.points.back();
            float predictedT = last.t;
            if (track.points.size() >= 2) {
                predictedT += last.t - track.points[track.points.size() - 2].t;
            }
            int best = -1;
            float bestDistance = std::numeric_limits<float>::max();
            for (int i = 0; i < static_cast<int>(peaks.size()); ++i) {
                const float distance = std::abs(peaks[i].t - predictedT);
                if (!used[i] && distance < bestDistance) {
                    bestDistance = distance;
                    best = i;
                }
            }
            if (best < 0 || bestDistance > config_.maxMatchDistance) {
                if (++track.missing >= config_.maxMissingSlices) track.active = false;
                continue;
            }
            if (last.radius > 1e-6f &&
                peaks[best].radius > last.radius * config_.radiusIncreaseRatio) {
                track.active = false;
                track.stoppedAtMerge = true;
                continue;
            }
            track.points.push_back(peaks[best]);
            track.missing = 0;
            used[best] = true;
        }
    }

    std::vector<BilletBendResult> results;
    for (const Track &track : tracks) {
        BilletBendResult result;
        result.id = track.id;
        result.stoppedAtMerge = track.stoppedAtMerge;
        if (static_cast<int>(track.points.size()) < config_.minTrackPoints) {
            results.push_back(result);
            continue;
        }
        std::vector<cv::Point2f> centerPoints;
        std::vector<float> widths;
        for (const Peak &peak : track.points) {
            centerPoints.push_back(peak.point);
            widths.push_back(2.0f * peak.radius);
        }
        cv::fitLine(centerPoints, result.fittedLine, cv::DIST_HUBER, 0, 0.01, 0.01);
        std::vector<float> normalized;
        for (size_t i = 0; i < centerPoints.size(); ++i) {
            if (widths[i] >= 2.0f * config_.minRadius) {
                normalized.push_back(pointLineDistance(centerPoints[i], result.fittedLine) /
                                     widths[i]);
            }
        }
        if (normalized.empty()) {
            results.push_back(result);
            continue;
        }
        std::sort(normalized.begin(), normalized.end());
        const size_t p95 = static_cast<size_t>(std::floor(0.95 * (normalized.size() - 1)));
        const float squareSum = std::accumulate(normalized.begin(), normalized.end(), 0.0f,
                                                [](float sum, float v) { return sum + v * v; });
        result.valid = true;
        result.bend95 = normalized[p95];
        result.bendMax = normalized.back();
        result.bendRms = std::sqrt(squareSum / normalized.size());
        result.bent = result.bend95 > config_.bend95Threshold &&
                      result.bendRms > config_.bendRmsThreshold;
        result.medianWidth = median(widths);
        result.validLengthRatio = std::min(1.0f,
            (track.points.back().s - track.points.front().s) / std::max(totalLength, 1.0f));
        result.centerPoints = std::move(centerPoints);
        results.push_back(std::move(result));
    }
    return results;
}

void BilletBendDetector::draw(cv::Mat &image, const std::vector<BilletBendResult> &results) {
    if (image.empty()) return;
    for (const BilletBendResult &result : results) {
        if (!result.valid) continue;
        const cv::Scalar color = result.bent ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        for (const cv::Point2f &point : result.centerPoints) {
            cv::circle(image, point, 2, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        }
        const float extent = static_cast<float>(std::max(image.cols, image.rows)) * 2.0f;
        const cv::Point2f origin(result.fittedLine[2], result.fittedLine[3]);
        const cv::Point2f direction(result.fittedLine[0], result.fittedLine[1]);
        cv::line(image, origin - direction * extent, origin + direction * extent,
                 color, 2, cv::LINE_AA);
        const cv::Point anchor = result.centerPoints.empty() ? cv::Point(10, 20) :
                                 cv::Point(cvRound(result.centerPoints.front().x),
                                           cvRound(result.centerPoints.front().y));
        const std::string label = cv::format("ID:%d B95:%.3f RMS:%.3f%s", result.id,
                                             result.bend95, result.bendRms,
                                             result.stoppedAtMerge ? " MERGE" : "");
        cv::putText(image, label, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
    }
}

} // namespace JHDeepCore
