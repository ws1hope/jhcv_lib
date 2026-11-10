#include "tiangangfujian_pipeline.h"

#include <algorithm>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

TiangangfujianPipeline::TiangangfujianPipeline(const TiangangfujianServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.det_model, "", dev_id);
    std::cout << "[OK] Tiangangfujian section det model loaded: " << config_.det_model << std::endl;

    // iou=0.9：与 tiebiao 的 char_seg_ 一致，宽松保留全部字符候选框，
    // 避免密集字符在 NMS 阶段被误去重。
    char_seg_ = std::make_unique<InstanceSegmenter>(config_.char_seg_model, "", dev_id, "",
                                                     0.25f, 0.9f);
    std::cout << "[OK] Tiangangfujian char seg model loaded: " << config_.char_seg_model << std::endl;

    direction_cls_ = std::make_unique<Classifier>(config_.direction_cls_model, "", dev_id);
    std::cout << "[OK] Tiangangfujian direction cls model loaded: " << config_.direction_cls_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] Tiangangfujian OCR model loaded: " << config_.ocr_model << std::endl;

    warmup();
}

void TiangangfujianPipeline::warmup()
{
    std::cout << "[INFO] Warming up tiangangfujian models..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det_results;
    det_->process(dummy, det_results);
    std::cout << "[OK] Section detector warmed up" << std::endl;

    std::vector<InstanceSegmentationResult> seg_results;
    char_seg_->process(dummy, seg_results);
    std::cout << "[OK] Char seg warmed up" << std::endl;

    std::vector<ClassificationResult> cls_results;
    direction_cls_->process(dummy, cls_results);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(dummy, ocr_results);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

TiangangfujianPipelineResult TiangangfujianPipeline::process(const cv::Mat& image, bool verbose)
{
    TiangangfujianPipelineResult result;
    std::vector<TiangangfujianSectionResult> sections;

    if (config_.rois.empty()) {
        // 未配置 ROI：整图作为一个区域处理
        if (verbose) std::cout << "[DEBUG] No rois configured, process whole image" << std::endl;
        processRegion(image, cv::Point(0, 0), 0, verbose, sections);
    } else {
        // 全局固定 ROI 列表：逐区域裁剪后独立处理
        if (verbose) std::cout << "[DEBUG] " << config_.rois.size() << " roi(s) configured" << std::endl;
        for (int i = 0; i < (int)config_.rois.size(); i++) {
            const auto& r = config_.rois[i];
            cv::Rect roi = ImageHelper::safeClampROI(
                r[0], r[1], r[2], r[3], image.cols, image.rows);
            if (roi.area() <= 0) {
                if (verbose) std::cout << "[DEBUG] roi" << (i + 1) << " invalid, skip" << std::endl;
                continue;
            }
            if (verbose) {
                std::cout << "[DEBUG] roi" << (i + 1) << " = (" << roi.x << "," << roi.y
                          << "," << roi.width << "," << roi.height << ")" << std::endl;
            }
            cv::Mat region = image(roi).clone();
            processRegion(region, roi.tl(), i + 1, verbose, sections);
        }
    }

    // 断面按原图 x 从左到右排序，拼接结果符合阅读顺序
    std::sort(sections.begin(), sections.end(),
        [](const TiangangfujianSectionResult& a, const TiangangfujianSectionResult& b) {
            return a.bbox_on_src.x < b.bbox_on_src.x;
        });

    result.sections = sections;

    std::string all_results;
    for (auto& sec : sections) {
        if (!sec.ocr_text.empty()) all_results += sec.ocr_text;
    }
    result.all_results = all_results;
    result.annotated_image = createAnnotatedImage(image, result.sections);

    return result;
}

void TiangangfujianPipeline::processRegion(const cv::Mat& region_img,
                                            const cv::Point& region_offset,
                                            int roi_index,
                                            bool verbose,
                                            std::vector<TiangangfujianSectionResult>& sections)
{
    std::vector<cv::Mat> det_imgs = {region_img};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_result = det_results.empty() ? DetectionResult{} : det_results[0];

    if (verbose) {
        std::cout << "[DEBUG] region roi" << roi_index
                  << " sections detected: " << det_result.num_detections << std::endl;
    }

    std::vector<Detection> dets = det_result.detections;
    // 区域内断面按 x 从左到右
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.x < b.bbox.x;
        });

    for (int i = 0; i < (int)dets.size(); i++) {
        const auto& d = dets[i];

        if (verbose) {
            std::cout << "  sec[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }

        cv::Rect roi = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            region_img.cols, region_img.rows);
        if (roi.area() <= 0) continue;

        TiangangfujianSectionResult sec;
        sec.roi_index = roi_index;
        sec.class_name = d.class_name;
        sec.confidence = d.confidence;
        sec.class_id = d.class_id;
        sec.bbox_on_src = cv::Rect(roi.tl() + region_offset, roi.br() + region_offset);

        cv::Mat section_img = region_img(roi).clone();
        if (section_img.empty()) continue;
        sec.section_image = section_img;

        // 字符实例分割 + 断面角度矫正
        cv::Mat rotated_image;
        std::vector<CharCropInfo> char_crops;
        detectAndRotateChars(section_img, rotated_image, char_crops);
        sec.rotated_image = rotated_image;

        if (char_crops.empty()) {
            if (verbose) std::cout << "  sec[" << i << "] no chars detected" << std::endl;
            sections.push_back(std::move(sec));
            continue;
        }

        if (verbose) std::cout << "  sec[" << i << "] "
                               << char_crops.size() << " chars detected" << std::endl;

        // 按矫正后 y 坐标从小到大排序（从上到下阅读顺序）
        std::sort(char_crops.begin(), char_crops.end(),
            [](const CharCropInfo& a, const CharCropInfo& b) {
                return a.center_y < b.center_y;
            });

        // 方向分类：断面内全部字符投票，多数为 180 则整断面统一翻转（同 tiebiao）
        std::vector<cv::Mat> char_images_before;
        for (auto& crop : char_crops) {
            char_images_before.push_back(crop.image.clone());
        }
        int dir_flag = classifyDirection(char_images_before);
        sec.direction_flag = dir_flag;
        if (verbose) std::cout << "  sec[" << i << "] direction: " << dir_flag
                               << (dir_flag == 180 ? " (flip 180)" : "") << std::endl;

        for (int ci = 0; ci < (int)char_crops.size(); ci++) {
            if (dir_flag == 180 && !char_crops[ci].image.empty()) {
                cv::Mat flipped;
                cv::flip(char_crops[ci].image, flipped, -1);
                char_crops[ci].image = flipped.clone();
            }
        }

        // 逐字符 OCR，按排序顺序拼接
        std::string text;
        for (int ci = 0; ci < (int)char_crops.size(); ci++) {
            TiangangfujianCharResult chr;
            chr.bbox_on_rotated = char_crops[ci].bbox;
            chr.image_before_flip = char_images_before[ci];
            chr.image_after_flip = char_crops[ci].image;
            chr.direction_flag = dir_flag;

            cv::Mat bgr;
            if (char_crops[ci].image.channels() == 1)
                cv::cvtColor(char_crops[ci].image, bgr, cv::COLOR_GRAY2BGR);
            else
                bgr = char_crops[ci].image;

            std::vector<cv::Mat> ocr_imgs = {bgr};
            std::vector<OCRResult> ocr_results;
            ocr_->process(ocr_imgs, ocr_results);

            std::string t;
            if (!ocr_results.empty() && !ocr_results[0].boxes.empty()) {
                t = ocr_results[0].boxes[0].text;
            }
            chr.ocr_text = t;
            sec.chars.push_back(std::move(chr));

            text += t;
            if (verbose) {
                std::cout << "    char[" << ci << "] text=\"" << t << "\"" << std::endl;
            }
        }

        sec.ocr_text = text;
        sections.push_back(std::move(sec));
    }
}

void TiangangfujianPipeline::detectAndRotateChars(const cv::Mat& section_roi,
                                                   cv::Mat& rotated_image,
                                                   std::vector<CharCropInfo>& char_crops)
{
    char_crops.clear();

    std::vector<cv::Mat> imgs = {section_roi};
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
            section_roi.cols, section_roi.rows);
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

    // 取高度最大字符的角度作为整断面矫正角度
    ImageHelper::sortByHeight(char_infos);
    float angle = char_infos.back().angle;

    int sum_x = 0, sum_y = 0;
    for (auto& info : char_infos) {
        sum_x += info.center_x + info.bbox_offset.x;
        sum_y += info.center_y + info.bbox_offset.y;
    }
    int center_x = sum_x / static_cast<int>(char_infos.size());
    int center_y = sum_y / static_cast<int>(char_infos.size());

    // bias 边距：旋转后角点可能越出原断面范围，四周补 30 像素黑边
    const int bias = 30;
    cv::Mat padded = cv::Mat::zeros(
        section_roi.rows + 2 * bias, section_roi.cols + 2 * bias, CV_8UC3);
    section_roi.copyTo(padded(cv::Rect(bias, bias, section_roi.cols, section_roi.rows)));

    int x_off = 0, y_off = 0;
    rotated_image = ImageHelper::rotateImageAroundPoint(
        padded, center_x + bias, center_y + bias, angle, x_off, y_off);

    char_crops = ImageHelper::rotateAndRemapBBoxes(
        char_infos, cv::Point2f(static_cast<float>(center_x),
                                 static_cast<float>(center_y)),
        angle, rotated_image, bias, bias, x_off, y_off);
}

int TiangangfujianPipeline::classifyDirection(const std::vector<cv::Mat>& char_images)
{
    if (char_images.empty()) return 0;

    int count_0 = 0, count_180 = 0;
    float conf_0_sum = 0.f;     // 投"0/正向"票的置信度之和
    float conf_180_sum = 0.f;   // 投"180/翻转"票的置信度之和
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
            if (results[0].class_id == 0) {
                count_0++;
                conf_0_sum += results[0].confidence;
            }
            else {
                count_180++;
                conf_180_sum += results[0].confidence;
            }
        }
    }

    // 阈值过滤仅当 dir_flip_conf_threshold > 0 时生效；
    // 默认 0 = 不过滤，保持与 tiebiao 一致的行为。
    //   多数票为翻转(180)：平均置信度 < 阈值 -> 不翻转
    //   多数票为正向(0)：平均置信度  < 阈值 -> 反向翻转(180)
    float thr = config_.dir_flip_conf_threshold;
    if (count_180 > count_0) {
        if (thr > 0.f && (conf_180_sum / count_180) < thr) return 0;
        return 180;
    }
    if (count_0 > count_180) {
        if (thr > 0.f && (conf_0_sum / count_0) < thr) return 180;
        return 0;
    }
    return 0;
}

cv::Mat TiangangfujianPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<TiangangfujianSectionResult>& sections)
{
    cv::Mat annotated = src_img.clone();
    int text_y = 70;
    for (int i = 0; i < (int)sections.size(); i++) {
        const auto& s = sections[i];
        cv::rectangle(annotated, s.bbox_on_src, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("sec%d %s %.2f", i + 1,
                                       s.class_name.c_str(), s.confidence);
        cv::putText(annotated, label,
                    cv::Point(s.bbox_on_src.x, s.bbox_on_src.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (!s.ocr_text.empty()) {
            std::string disp = cv::format("sec%d: %s", i + 1, s.ocr_text.c_str());
            cv::putText(annotated, disp, cv::Point(10, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
            text_y += 75;
        }
    }
    return annotated;
}

} // namespace Pipeline
} // namespace JHDeepCore
