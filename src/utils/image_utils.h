#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define IMAGE_UTILS_PI 3.14159265358979323846

struct MinAreaRectResult {
    float angle = 0.0f;
    int center_x = 0;
    int center_y = 0;
    cv::Point2f corners[4];
    float height = 0.0f;
};

struct CharAngleInfo {
    float angle = 0.0f;
    int center_x = 0;
    int center_y = 0;
    int class_id = 0;
    cv::Point2f corners[4];
    cv::Point2i bbox_offset;
    float height = 0.0f;
};

struct CharCropInfo {
    int center_x = 0;
    int center_y = 0;
    cv::Mat image;
};

class ImageHelper {
public:
    ImageHelper() = delete;

    static MinAreaRectResult computeMinAreaRect(const cv::Mat& mask);

    static cv::Mat rotateImageAroundPoint(const cv::Mat& img,
                                           int center_x, int center_y,
                                           float angle,
                                           int& x_offset, int& y_offset);

    static cv::Point2f rotatePoint(const cv::Point2f& p,
                                    const cv::Point2f& center,
                                    float angle);

    static cv::Rect safeClampROI(int x, int y, int w, int h,
                                  int img_w, int img_h);

    static std::vector<CharCropInfo> rotateAndRemapBBoxes(
        std::vector<CharAngleInfo>& char_infos,
        const cv::Point2f& rotation_center,
        float angle,
        const cv::Mat& rotated_image,
        int bias_w, int bias_h,
        int x_offset, int y_offset);

    static void sortByHeight(std::vector<CharAngleInfo>& infos);

private:
    static cv::Mat translateImage(const cv::Mat& src, int x_offset, int y_offset);
};

void debugSave(const std::string& tag, const cv::Mat& img);
