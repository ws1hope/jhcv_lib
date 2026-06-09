#include "image_utils.h"
#include "file_utils.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

static double calculateAngle(const cv::Point& p1, const cv::Point& p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    if (std::abs(dx) < 1e-9) return 90.0;
    double k = dy / dx;
    return std::atan(k) * 180.0 / IMAGE_UTILS_PI;
}

MinAreaRectResult ImageHelper::computeMinAreaRect(const cv::Mat& mask)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    MinAreaRectResult result;
    if (contours.empty()) return result;

    struct RectInfo {
        float angle;
        int center_x, center_y;
        cv::Point2f corners[4];
        float height;
    };

    std::vector<RectInfo> rects;
    for (auto& contour : contours) {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f pts[4];
        rect.points(pts);

        RectInfo ri;
        for (int j = 0; j < 4; j++) ri.corners[j] = pts[j];

        double dw = std::sqrt(std::pow(pts[0].x - pts[1].x, 2) +
                              std::pow(pts[0].y - pts[1].y, 2));
        double dh = std::sqrt(std::pow(pts[1].x - pts[2].x, 2) +
                              std::pow(pts[1].y - pts[2].y, 2));

        if (dw > dh) {
            ri.angle = static_cast<float>(calculateAngle(
                cv::Point(pts[0].x, pts[0].y), cv::Point(pts[1].x, pts[1].y)));
            ri.height = static_cast<float>(dw);
        } else {
            ri.angle = static_cast<float>(calculateAngle(
                cv::Point(pts[1].x, pts[1].y), cv::Point(pts[2].x, pts[2].y)));
            ri.height = static_cast<float>(dh);
        }

        ri.center_x = static_cast<int>(rect.center.x);
        ri.center_y = static_cast<int>(rect.center.y);
        rects.push_back(ri);
    }

    std::sort(rects.begin(), rects.end(),
              [](const RectInfo& a, const RectInfo& b) { return a.height > b.height; });

    result.angle = rects[0].angle;
    result.center_x = rects[0].center_x;
    result.center_y = rects[0].center_y;
    result.height = rects[0].height;
    for (int j = 0; j < 4; j++) result.corners[j] = rects[0].corners[j];
    return result;
}

cv::Mat ImageHelper::translateImage(const cv::Mat& src, int x_offset, int y_offset)
{
    cv::Mat result = cv::Mat::zeros(src.size(), src.type());

    cv::Rect src_rect, dst_rect;
    if (x_offset >= 0) {
        src_rect = cv::Rect(0, 0, src.cols - x_offset, src.rows);
        dst_rect = cv::Rect(x_offset, 0, src.cols - x_offset, src.rows);
    } else {
        src_rect = cv::Rect(-x_offset, 0, src.cols + x_offset, src.rows);
        dst_rect = cv::Rect(0, 0, src.cols + x_offset, src.rows);
    }

    if (y_offset >= 0) {
        src_rect &= cv::Rect(0, 0, src.cols, src.rows - y_offset);
        dst_rect &= cv::Rect(0, y_offset, src.cols, src.rows - y_offset);
    } else {
        src_rect &= cv::Rect(0, -y_offset, src.cols, src.rows + y_offset);
        dst_rect &= cv::Rect(0, 0, src.cols, src.rows + y_offset);
    }

    int rw = std::min(src_rect.width, dst_rect.width);
    int rh = std::min(src_rect.height, dst_rect.height);
    if (rw <= 0 || rh <= 0) return result;

    src_rect.width = rw; src_rect.height = rh;
    dst_rect.width = rw; dst_rect.height = rh;

    if (src_rect.x >= 0 && src_rect.y >= 0 &&
        dst_rect.x >= 0 && dst_rect.y >= 0 &&
        src_rect.x + rw <= src.cols && src_rect.y + rh <= src.rows &&
        dst_rect.x + rw <= result.cols && dst_rect.y + rh <= result.rows) {
        src(src_rect).copyTo(result(dst_rect));
    }
    return result;
}

cv::Mat ImageHelper::rotateImageAroundPoint(const cv::Mat& img,
                                             int center_x, int center_y,
                                             float angle,
                                             int& x_offset, int& y_offset)
{
    cv::Point2f center_img(img.cols / 2.0f, img.rows / 2.0f);
    cv::Point2f center_ocr(static_cast<float>(center_x), static_cast<float>(center_y));

    x_offset = static_cast<int>(center_img.x - center_ocr.x);
    y_offset = static_cast<int>(center_img.y - center_ocr.y);

    cv::Mat translated = translateImage(img, x_offset, y_offset);

    cv::Mat rot_mat = cv::getRotationMatrix2D(center_img, static_cast<double>(angle), 1.0);
    cv::Mat rotated;
    cv::warpAffine(translated, rotated, rot_mat, translated.size());
    return rotated;
}

cv::Point2f ImageHelper::rotatePoint(const cv::Point2f& p,
                                      const cv::Point2f& center,
                                      float angle)
{
    float rad = angle * static_cast<float>(IMAGE_UTILS_PI) / 180.0f;
    float cos_v = std::cos(rad);
    float sin_v = std::sin(rad);
    cv::Point2f result;
    result.x = (p.x - center.x) * cos_v - (p.y - center.y) * sin_v + center.x;
    result.y = (p.x - center.x) * sin_v + (p.y - center.y) * cos_v + center.y;
    return result;
}

cv::Rect ImageHelper::safeClampROI(int x, int y, int w, int h,
                                    int img_w, int img_h)
{
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x >= img_w || y >= img_h) return cv::Rect(0, 0, 0, 0);
    if (x + w > img_w) w = img_w - x;
    if (y + h > img_h) h = img_h - y;
    if (w <= 0 || h <= 0) return cv::Rect(0, 0, 0, 0);
    return cv::Rect(x, y, w, h);
}

std::vector<CharCropInfo> ImageHelper::rotateAndRemapBBoxes(
    std::vector<CharAngleInfo>& char_infos,
    const cv::Point2f& rotation_center,
    float angle,
    const cv::Mat& rotated_image,
    int bias_w, int bias_h,
    int x_offset, int y_offset)
{
    std::vector<CharCropInfo> crops;

    for (auto& info : char_infos) {
        std::vector<float> pts_x, pts_y;
        cv::Point2f center(rotation_center.x + bias_w, rotation_center.y + bias_h);

        for (int k = 0; k < 4; k++) {
            cv::Point2f orig(info.corners[k].x + info.bbox_offset.x + bias_w,
                             info.corners[k].y + info.bbox_offset.y + bias_h);
            cv::Point2f rotated = rotatePoint(orig, center, -angle);
            pts_x.push_back(rotated.x + x_offset);
            pts_y.push_back(rotated.y + y_offset);
        }

        int min_x = static_cast<int>(*std::min_element(pts_x.begin(), pts_x.end()));
        int max_x = static_cast<int>(*std::max_element(pts_x.begin(), pts_x.end()));
        int min_y = static_cast<int>(*std::min_element(pts_y.begin(), pts_y.end()));
        int max_y = static_cast<int>(*std::max_element(pts_y.begin(), pts_y.end()));

        cv::Rect roi = safeClampROI(min_x, min_y,
                                     max_x - min_x, max_y - min_y,
                                     rotated_image.cols, rotated_image.rows);
        if (roi.area() <= 0) continue;

        cv::Mat crop = rotated_image(roi).clone();
        if (crop.empty()) continue;

        CharCropInfo crop_info;
        crop_info.center_x = (max_x + min_x) / 2;
        crop_info.center_y = (max_y + min_y) / 2;
        crop_info.bbox = roi;
        crop_info.image = crop;
        crops.push_back(crop_info);
    }

    return crops;
}

void ImageHelper::sortByHeight(std::vector<CharAngleInfo>& infos)
{
    std::sort(infos.begin(), infos.end(),
              [](const CharAngleInfo& a, const CharAngleInfo& b) {
                  return a.height < b.height;
              });
}

void debugSave(const std::string& tag, const cv::Mat& img)
{
    if (img.empty()) return;
    FileHelper::ensureDirectoryExists("debug");

    time_t now = time(nullptr);
    struct tm t;
#ifdef _WIN32
    localtime_s(&t, &now);
#else
    localtime_r(&now, &t);
#endif

    std::ostringstream oss;
    oss << "debug/" << tag
        << "_" << std::setfill('0')
        << (t.tm_year + 1900)
        << std::setw(2) << (t.tm_mon + 1)
        << std::setw(2) << t.tm_mday
        << std::setw(2) << t.tm_hour
        << std::setw(2) << t.tm_min
        << std::setw(2) << t.tm_sec
        << ".jpg";

    cv::imwrite(oss.str(), img);
}
