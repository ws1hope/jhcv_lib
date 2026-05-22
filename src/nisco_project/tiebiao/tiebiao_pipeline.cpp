#include "tiebiao_pipeline.h"
#include "infer_utils.h"
#include "file_utils.h"

#include <chrono>
#include <ctime>
#include <iostream>
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

TiebiaoPipeline::TiebiaoPipeline(const TiebiaoConfig& config)
    : config_(config)
{
    int dev_id = (config.device == "cuda" || config.device == "gpu") ? 0 : -1;

    label_seg_ = std::make_unique<InstanceSegmenter>(config.label_seg_model, "", dev_id);
    std::cout << "[OK] Label seg model loaded: " << config.label_seg_model << std::endl;

    char_seg_ = std::make_unique<InstanceSegmenter>(config.char_seg_model, "", dev_id);
    std::cout << "[OK] Char seg model loaded: " << config.char_seg_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config.ocr_model, config.ocr_label, dev_id);
    std::cout << "[OK] OCR model loaded: " << config.ocr_model << std::endl;

    direction_cls_ = std::make_unique<Classifier>(config.direction_cls_model, "", dev_id);
    std::cout << "[OK] Direction cls model loaded: " << config.direction_cls_model << std::endl;

    warmup();
}

void TiebiaoPipeline::warmup()
{
    std::cout << "[INFO] Warming up models..." << std::endl;
    cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Mat> imgs = {dummy};

    std::vector<InstanceSegmentationResult> seg_results;
    label_seg_->process(imgs, seg_results);
    std::cout << "[OK] Label seg warmed up" << std::endl;

    char_seg_->process(imgs, seg_results);
    std::cout << "[OK] Char seg warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(imgs, ocr_results);
    std::cout << "[OK] OCR warmed up" << std::endl;

    std::vector<ClassificationResult> cls_results;
    direction_cls_->process(imgs, cls_results);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

std::vector<std::pair<cv::Mat, int>> TiebiaoPipeline::detectLabels(const cv::Mat& image)
{
    std::vector<cv::Mat> imgs = {image};
    std::vector<InstanceSegmentationResult> results;
    label_seg_->process(imgs, results);

    std::vector<std::pair<cv::Mat, int>> label_rois;
    if (results.empty() || results[0].num_detections <= 0) return label_rois;

    auto& det = results[0];
    for (int i = 0; i < det.num_detections; i++) {
        auto& d = det.detections[i];
        auto& mask = det.masks[i];

        cv::Rect safe = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            image.cols, image.rows);
        if (safe.area() <= 0) continue;

        cv::Mat roi = image(safe).clone();
        label_rois.emplace_back(roi, d.class_id);
    }

    return label_rois;
}

void TiebiaoPipeline::detectAndRotateChars(const cv::Mat& label_roi,
                                             cv::Mat& rotated_image,
                                             std::vector<CharCropInfo>& char_crops)
{
    char_crops.clear();

    std::vector<cv::Mat> imgs = {label_roi};
    std::vector<InstanceSegmentationResult> results;
    char_seg_->process(imgs, results);

    if (results.empty() || results[0].num_detections <= 0) return;

    auto& det = results[0];
    std::vector<CharAngleInfo> char_infos;

    for (int i = 0; i < det.num_detections; i++) {
        auto& d = det.detections[i];
        auto& mask_full = det.masks[i];

        cv::Rect safe = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            label_roi.cols, label_roi.rows);
        if (safe.area() <= 0) continue;

        cv::Mat mask_crop = mask_full(safe);
        MinAreaRectResult mr = ImageHelper::computeMinAreaRect(mask_crop);

        CharAngleInfo info;
        info.angle = mr.angle;
        info.center_x = mr.center_x;
        info.center_y = mr.center_y;
        info.class_id = d.class_id;
        info.bbox_offset = cv::Point2i(d.bbox.x, d.bbox.y);
        info.height = mr.height;
        for (int j = 0; j < 4; j++) info.corners[j] = mr.corners[j];
        char_infos.push_back(info);
    }

    if (char_infos.empty()) return;

    ImageHelper::sortByHeight(char_infos);
    float angle = char_infos.back().angle;

    int sum_x = 0, sum_y = 0;
    for (auto& info : char_infos) {
        sum_x += info.center_x + info.bbox_offset.x;
        sum_y += info.center_y + info.bbox_offset.y;
    }
    int center_x = sum_x / static_cast<int>(char_infos.size());
    int center_y = sum_y / static_cast<int>(char_infos.size());

    const int bias = 30;
    cv::Mat padded = cv::Mat::zeros(
        label_roi.rows + 2 * bias, label_roi.cols + 2 * bias, CV_8UC3);
    label_roi.copyTo(padded(cv::Rect(bias, bias, label_roi.cols, label_roi.rows)));

    int x_off = 0, y_off = 0;
    rotated_image = ImageHelper::rotateImageAroundPoint(
        padded, center_x + bias, center_y + bias, angle, x_off, y_off);

    char_crops = ImageHelper::rotateAndRemapBBoxes(
        char_infos, cv::Point2f(static_cast<float>(center_x),
                                 static_cast<float>(center_y)),
        angle, rotated_image, bias, bias, x_off, y_off);
}

int TiebiaoPipeline::classifyDirection(const std::vector<cv::Mat>& char_images)
{
    if (char_images.empty()) return 0;

    int count_0 = 0, count_180 = 0;
    for (auto& img : char_images) {
        if (img.empty()) continue;
        cv::Mat bgr;
        if (img.channels() == 1)
            cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        else
            bgr = img;

        std::vector<cv::Mat> imgs = {bgr};
        std::vector<ClassificationResult> results;
        direction_cls_->process(imgs, results);

        if (!results.empty()) {
            if (results[0].class_id == 0)
                count_0++;
            else
                count_180++;
        }
    }

    return (count_0 >= count_180) ? 0 : 180;
}

std::vector<std::string> TiebiaoPipeline::recognizeChars(const std::vector<CharCropInfo>& crops)
{
    std::vector<std::string> texts;
    for (auto& crop : crops) {
        if (crop.image.empty()) {
            texts.push_back("");
            continue;
        }

        cv::Mat bgr;
        if (crop.image.channels() == 1)
            cv::cvtColor(crop.image, bgr, cv::COLOR_GRAY2BGR);
        else
            bgr = crop.image;

        std::vector<cv::Mat> imgs = {bgr};
        std::vector<OCRResult> results;
        ocr_->process(imgs, results);

        std::string text;
        if (!results.empty() && !results[0].boxes.empty()) {
            text = results[0].boxes[0].text;
        }
        texts.push_back(text);
    }
    return texts;
}

cv::Mat TiebiaoPipeline::createAnnotatedImage(const cv::Mat& src_img,
                                                const cv::Mat& rotated_img,
                                                const std::vector<std::string>& ocr_texts,
                                                const std::vector<CharCropInfo>& crops,
                                                const std::string& label_type,
                                                int direction_flag)
{
    cv::Mat src_copy = src_img.clone();

    cv::Mat yuanbiao = rotated_img.clone();
    if (direction_flag == 180) {
        cv::flip(yuanbiao, yuanbiao, -1);
    }

    double scale = 3.0;
    cv::Mat yuanbiao_scaled;
    cv::resize(yuanbiao, yuanbiao_scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);

    int x = src_copy.cols - yuanbiao_scaled.cols - 1000;
    int y = 20;
    if (x < 0) x = 0;

    cv::Rect dst_rect(x, y, yuanbiao_scaled.cols, yuanbiao_scaled.rows);
    cv::Rect img_rect(0, 0, src_copy.cols, src_copy.rows);
    cv::Rect clipped = dst_rect & img_rect;

    if (!clipped.empty()) {
        cv::Mat src_roi = yuanbiao_scaled(
            cv::Rect(clipped.x - x, clipped.y - y, clipped.width, clipped.height));
        src_roi.copyTo(src_copy(clipped));
    }

    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.5;
    cv::Scalar color_red(0, 0, 255);
    cv::Scalar color_blue(255, 0, 0);
    int thickness = 3;
    int text_y = yuanbiao_scaled.rows + 60;

    cv::putText(src_copy, label_type, cv::Point(50, text_y),
                font_face, font_scale, color_blue, thickness);

    for (int i = 0; i < (int)ocr_texts.size(); i++) {
        if (ocr_texts[i].empty()) continue;
        std::string line = ocr_texts[i];
        text_y += 50;
        cv::putText(src_copy, line, cv::Point(50, text_y),
                    font_face, font_scale, color_red, thickness);
    }

    return src_copy;
}

TiebiaoResult TiebiaoPipeline::process(const cv::Mat& image,
                                         int station_id,
                                         const std::string& heat_number,
                                         bool verbose)
{
    auto infer_start = std::chrono::high_resolution_clock::now();

    TiebiaoResult result;
    result.picture_id = 0;

    auto labels = detectLabels(image);
    if (labels.empty()) {
        result.state_flag = "NG";
        if (verbose) std::cout << "[DEBUG] No labels detected" << std::endl;
        return result;
    }

    if (verbose) std::cout << "[DEBUG] Detected " << labels.size() << " labels" << std::endl;

    std::string ocr_combined;
    cv::Mat best_annotated;

    for (int kk = 0; kk < (int)labels.size(); kk++) {
        auto& [label_roi, class_id] = labels[kk];
        std::string label_type = (class_id == 0) ? "circle" : "hexagon";

        if (verbose) std::cout << "[DEBUG] Label " << kk << " type=" << label_type << std::endl;

        cv::Mat rotated_image;
        std::vector<CharCropInfo> char_crops;
        detectAndRotateChars(label_roi, rotated_image, char_crops);

        if (char_crops.empty()) {
            if (verbose) std::cout << "[DEBUG] No chars detected in label " << kk << std::endl;
            continue;
        }

        if (verbose) std::cout << "[DEBUG] " << char_crops.size() << " chars detected" << std::endl;

        std::vector<cv::Mat> char_images;
        for (auto& crop : char_crops) {
            char_images.push_back(crop.image);
        }
        int dir_flag = classifyDirection(char_images);

        if (verbose) std::cout << "[DEBUG] Direction: " << dir_flag << std::endl;

        std::vector<std::string> ocr_texts = recognizeChars(char_crops);

        if (verbose) {
            for (int i = 0; i < (int)ocr_texts.size(); i++) {
                std::cout << "[DEBUG]   char[" << i << "] text=\"" << ocr_texts[i] << "\"" << std::endl;
            }
        }

        int best_idx = InferHelper::findBestLuhaoMatch(ocr_texts, heat_number);

        std::string luhao;
        if (best_idx >= 0) {
            luhao = InferHelper::fixLuhaoChars(ocr_texts[best_idx]);
            if (verbose) std::cout << "[DEBUG] Best luhao match idx=" << best_idx
                                    << " text=\"" << luhao << "\"" << std::endl;
        } else {
            for (int i = 0; i < (int)ocr_texts.size(); i++) {
                if (ocr_texts[i].size() > 7) {
                    luhao = InferHelper::fixLuhaoChars(ocr_texts[i]);
                    break;
                }
            }
        }

        if (!luhao.empty()) {
            if (!ocr_combined.empty()) ocr_combined += ",";
            ocr_combined += label_type + "#" + luhao;
        }

        best_annotated = createAnnotatedImage(
            image, rotated_image, ocr_texts, char_crops, label_type, dir_flag);
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) std::cout << "[DEBUG] Total inference time: " << infer_ms << " ms" << std::endl;

    if (!ocr_combined.empty()) {
        result.state_flag = "OK";
        result.ocr_text = ocr_combined;
        result.annotated_image = best_annotated;
    } else {
        result.state_flag = "NG";
    }

    return result;
}

} // namespace Pipeline
} // namespace JHDeepCore
