#include "JHDeepCore.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace JHDeepCore {

class HomographyPrivate {
public:
    HomographyPrivate() = default;

    cv::Mat compute(const std::vector<PointPair> &pairs) {
        radarP_.clear();
        imgP_.clear();
        for (const auto &p : pairs) {
            radarP_.push_back(p.first);
            imgP_.push_back(p.second);
        }
        if (radarP_.size() < 4 || imgP_.size() != radarP_.size()) {
            return {};
        }
        matrix_ = cv::findHomography(radarP_, imgP_);
        return matrix_.clone();
    }

    std::vector<double> compute_flat(const std::vector<PointPair> &pairs) {
        std::vector<double> flat;
        cv::Mat mat = compute(pairs);
        if (mat.empty()) {
            return flat;
        }
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                flat.push_back(mat.at<double>(i, j));
            }
        }
        return flat;
    }

    void set_matrix(const std::vector<double> &values) {
        matrix_.release();
        cv::Mat tmp(values);
        matrix_ = tmp.reshape(1, 3).clone();
    }

    void set_matrix(const cv::Mat &mat) {
        matrix_ = mat.clone();
    }

    std::vector<cv::Point2f> project_points(const std::vector<cv::Point2f> &src) {
        std::vector<cv::Point2f> dst;
        if (matrix_.empty()) {
            return dst;
        }
        cv::perspectiveTransform(src, dst, matrix_);
        return dst;
    }

    cv::Point2f project_point(const cv::Point2f &pt) {
        auto result = project_points({pt});
        return result.empty() ? cv::Point2f() : result[0];
    }

    cv::Mat get_matrix() const {
        return matrix_.clone();
    }

private:
    std::vector<cv::Point2f> radarP_;
    std::vector<cv::Point2f> imgP_;
    cv::Mat matrix_;
};

Homography::Homography() : m_pHandle(std::make_shared<HomographyPrivate>()) {}

Homography::~Homography() = default;

cv::Mat Homography::compute(const std::vector<PointPair> &pairs) {
    return m_pHandle->compute(pairs);
}

std::vector<double> Homography::compute_flat(const std::vector<PointPair> &pairs) {
    return m_pHandle->compute_flat(pairs);
}

void Homography::set_matrix(const std::vector<double> &values) {
    m_pHandle->set_matrix(values);
}

void Homography::set_matrix(const cv::Mat &mat) {
    m_pHandle->set_matrix(mat);
}

std::vector<cv::Point2f> Homography::project_points(const std::vector<cv::Point2f> &src) {
    return m_pHandle->project_points(src);
}

cv::Point2f Homography::project_point(const cv::Point2f &pt) {
    return m_pHandle->project_point(pt);
}

cv::Mat Homography::get_matrix() const {
    return m_pHandle->get_matrix();
}

} // namespace JHDeepCore
