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

namespace {

// 几何判向：倾斜矫正后，字符框按从上到下(center_y)排序，
// bbox.width 递减 -> 正向(0°)，递增 -> 反向(180°)。
// 用相邻对的递减/递增计数取多数，抗噪；并列默认正向(0°)。
int classifyDirectionByGeometry(const std::vector<CharCropInfo>& char_crops,
                                 int* dec_out = nullptr, int* inc_out = nullptr)
{
    int n = static_cast<int>(char_crops.size());
    if (n < 2) {
        if (dec_out) *dec_out = 0;
        if (inc_out) *inc_out = 0;
        return 0;
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return char_crops[a].center_y < char_crops[b].center_y;   // 从上到下
    });
    int dec = 0, inc = 0;
    for (int i = 1; i < n; i++) {
        int w_prev = char_crops[order[i - 1]].bbox.width;
        int w_cur  = char_crops[order[i]].bbox.width;
        if (w_cur < w_prev) dec++;
        else if (w_cur > w_prev) inc++;
    }
    if (dec_out) *dec_out = dec;
    if (inc_out) *inc_out = inc;
    return (inc > dec) ? 180 : 0;   // 递减->0°，递增->180°，并列->0°
}

} // namespace

TiebiaoPipeline::TiebiaoPipeline(const TiebiaoConfig& config)
    : config_(config)
{
    int dev_id = (config.device == "cuda" || config.device == "gpu") ? 0 : -1;

    label_seg_ = std::make_unique<InstanceSegmenter>(config.label_seg_model, "", dev_id);
    std::cout << "[OK] Label seg model loaded: " << config.label_seg_model << std::endl;

    // iou=0.9：与 gx_jingzheng 的 seg_ 一致，宽松保留全部字符候选框，
    // 避免密集字符在 NMS 阶段被误去重。
    char_seg_ = std::make_unique<InstanceSegmenter>(config.char_seg_model, "", dev_id, "",
                                                     0.25f, 0.9f);
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
    // 默认 0 = 不过滤，保持原有行为（独立 tiebiao / dispatch 不受影响）。
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

cv::Mat TiebiaoPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<LabelDisplayInfo>& labels)
{
    const int margin = 15;
    const int row_h = 60;
    const int thumb_height = 200;
    const int char_margin = 4;
    const int gap = 15;
    const int section_gap = 10;
    const int pad_bottom = 15;
    const double strip_font_scale = 0.7;

    // 底部区域：每个标签一行，行内水平排列
    int bottom_area_h = 0;
    for (auto& lbl : labels) {
        int num_char_rows = (lbl.direction_flag == 180) ? 2 : 1;
        int strip_total_h = num_char_rows * (row_h + section_gap + 20);
        int row_total = std::max(thumb_height, strip_total_h);
        bottom_area_h += row_total + section_gap;
    }

    int canvas_h = src_img.rows + bottom_area_h + pad_bottom;
    cv::Mat result = cv::Mat::zeros(canvas_h, src_img.cols, src_img.type());
    src_img.copyTo(result(cv::Rect(0, 0, src_img.cols, src_img.rows)));

    // 左上角：最终匹配结果（大字体，从上到下排列）
    {
        int y = 70;
        int idx = 1;
        for (auto& lbl : labels) {
            std::string text = lbl.label_type + std::to_string(idx);
            if (!lbl.matched_luhao.empty()) text += ":" + lbl.matched_luhao;
            cv::putText(result, text, cv::Point(margin, y),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            y += 65;
            idx++;
        }
    }

    auto buildCharStrip = [&](const std::vector<cv::Mat>& imgs) -> cv::Mat {
        std::vector<cv::Mat> resized;
        for (auto& img : imgs) {
            if (img.empty()) continue;
            float s = static_cast<float>(row_h) / img.rows;
            cv::Mat r;
            cv::resize(img, r, cv::Size(), s, s, cv::INTER_LINEAR);
            resized.push_back(r);
        }
        if (resized.empty()) return cv::Mat();
        // 按显示宽度从大到小排列字符片段
        std::sort(resized.begin(), resized.end(),
            [](const cv::Mat& a, const cv::Mat& b) { return a.cols > b.cols; });
        int total_w = 0;
        for (auto& r : resized) total_w += r.cols + char_margin;
        total_w -= char_margin;
        cv::Mat strip = cv::Mat::zeros(row_h, total_w, CV_8UC3);
        int cx = 0;
        for (auto& r : resized) {
            cv::Rect roi(cx, 0, r.cols, r.rows);
            r.copyTo(strip(roi));
            cx += r.cols + char_margin;
        }
        return strip;
    };

    auto pasteAt = [&](const cv::Mat& img, int x, int y) {
        if (img.empty()) return;
        cv::Rect paste_rect(x, y, img.cols, img.rows);
        cv::Rect clipped = paste_rect & cv::Rect(0, 0, result.cols, result.rows);
        if (clipped.empty()) return;
        cv::Rect src_roi(clipped.x - x, clipped.y - y, clipped.width, clipped.height);
        img(src_roi).copyTo(result(clipped));
    };

    // 底部区域：每个标签一行（缩略图 + 字符片段水平排列）
    int bottom_y = src_img.rows;

    for (int lbl_idx = 0; lbl_idx < (int)labels.size(); lbl_idx++) {
        auto& lbl = labels[lbl_idx];
        std::string tag = lbl.label_type + std::to_string(lbl_idx + 1);

        cv::Mat tiebiao_img = lbl.rotated_image.clone();
        if (lbl.direction_flag == 180) {
            cv::flip(tiebiao_img, tiebiao_img, -1);
        }

        float thumb_scale = static_cast<float>(thumb_height) / tiebiao_img.rows;
        cv::Mat thumb;
        cv::resize(tiebiao_img, thumb, cv::Size(), thumb_scale, thumb_scale, cv::INTER_LINEAR);

        cv::Scalar color_green(0, 255, 0);
        for (auto& crop : lbl.char_crops) {
            cv::Rect draw_rect = crop.bbox;
            if (lbl.direction_flag == 180) {
                draw_rect.x = tiebiao_img.cols - draw_rect.x - draw_rect.width;
                draw_rect.y = tiebiao_img.rows - draw_rect.y - draw_rect.height;
            }
            cv::Rect scaled_rect(
                static_cast<int>(draw_rect.x * thumb_scale),
                static_cast<int>(draw_rect.y * thumb_scale),
                static_cast<int>(draw_rect.width * thumb_scale),
                static_cast<int>(draw_rect.height * thumb_scale));
            scaled_rect &= cv::Rect(0, 0, thumb.cols, thumb.rows);
            if (scaled_rect.area() > 0) {
                cv::rectangle(thumb, scaled_rect, color_green, 2);
            }
        }

        int cur_y = bottom_y;

        // 缩略图（左侧）
        pasteAt(thumb, margin, cur_y);
        int strip_x = margin + thumb.cols + gap;

        // 字符片段（缩略图右侧）
        if (lbl.direction_flag == 180) {
            cv::Mat strip_before = buildCharStrip(lbl.char_images_before_flip);
            cv::Mat strip_after = buildCharStrip(lbl.char_images_after_flip);

            int strip_y = cur_y + 20;
            cv::putText(result, tag + " Before flip:", cv::Point(strip_x, strip_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_before, strip_x, strip_y + 10);

            int strip2_y = strip_y + row_h + section_gap + 10;
            cv::putText(result, tag + " After flip:", cv::Point(strip_x, strip2_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_after, strip_x, strip2_y + 10);

            bottom_y = cur_y + thumb_height + section_gap;
        } else {
            cv::Mat strip_crops = buildCharStrip(lbl.char_images_after_flip);
            int strip_y = cur_y + (thumb_height - row_h) / 2;
            cv::putText(result, tag + " Char crops:", cv::Point(strip_x, strip_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_crops, strip_x, strip_y + 10);

            bottom_y = cur_y + thumb_height + section_gap;
        }
    }

    return result;
}

TiebiaoResult TiebiaoPipeline::process(const cv::Mat& image,
                                         int station_id,
                                         const std::string& heat_number,
                                         bool verbose)
{
    TiebiaoResult result;
    result.picture_id = 0;

    auto labels = detectLabels(image);
    if (labels.empty()) {
        result.state_flag = "NG";
        if (verbose) std::cout << "[DEBUG] No labels detected" << std::endl;
        return result;
    }

    if (verbose) std::cout << "[DEBUG] Detected " << labels.size() << " labels" << std::endl;

    return runLabels(image, labels, heat_number, verbose, result);
}

TiebiaoResult TiebiaoPipeline::process(const cv::Mat& image,
                                         int station_id,
                                         const std::string& heat_number,
                                         const std::vector<cv::Rect>& gangbiao_bboxes,
                                         bool verbose)
{
    TiebiaoResult result;
    result.picture_id = 0;

    // 复用 gx 已切好的 gangbiao bbox 作为 label ROI，跳过 label_seg_ 推理。
    // 每个 bbox 当作一个 circle 标签牌 (class_id=0)。
    std::vector<std::pair<cv::Mat, int>> labels;
    for (const auto& bb : gangbiao_bboxes) {
        cv::Rect safe = ImageHelper::safeClampROI(
            bb.x, bb.y, bb.width, bb.height, image.cols, image.rows);
        if (safe.area() <= 0) continue;
        labels.emplace_back(image(safe).clone(), 0);
    }

    if (labels.empty()) {
        result.state_flag = "NG";
        if (verbose) std::cout << "[DEBUG] No gangbiao bbox reused as label" << std::endl;
        return result;
    }

    if (verbose) std::cout << "[DEBUG] Reused " << labels.size()
                          << " gangbiao bbox(es) as labels (label_seg_ skipped)" << std::endl;

    return runLabels(image, labels, heat_number, verbose, result);
}

TiebiaoResult TiebiaoPipeline::runLabels(const cv::Mat& image,
                                          const std::vector<std::pair<cv::Mat, int>>& labels,
                                          const std::string& heat_number,
                                          bool verbose,
                                          TiebiaoResult& result)
{
    auto infer_start = std::chrono::high_resolution_clock::now();

    std::string ocr_combined;
    std::vector<LabelDisplayInfo> display_infos;

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

        int dec = 0, inc = 0;
        int dir_flag = classifyDirectionByGeometry(char_crops, &dec, &inc);
        if (verbose) std::cout << "[DEBUG] Direction: " << dir_flag
                               << " (width dec=" << dec << " inc=" << inc << ")" << std::endl;

        std::vector<cv::Mat> char_images_before;
        std::vector<cv::Mat> char_images_after;
        for (auto& crop : char_crops) {
            char_images_before.push_back(crop.image.clone());
            if (dir_flag == 180 && !crop.image.empty()) {
                cv::Mat flipped;
                cv::flip(crop.image, flipped, -1);
                char_images_after.push_back(flipped);
                crop.image = flipped.clone();
            } else {
                char_images_after.push_back(crop.image.clone());
            }
        }

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

        LabelDisplayInfo info;
        info.rotated_image = rotated_image;
        info.char_crops = char_crops;
        info.ocr_texts = ocr_texts;
        info.label_type = label_type;
        info.direction_flag = dir_flag;
        info.matched_luhao = luhao;
        info.char_images_before_flip = std::move(char_images_before);
        info.char_images_after_flip = std::move(char_images_after);
        display_infos.push_back(std::move(info));
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) std::cout << "[DEBUG] Total inference time: " << infer_ms << " ms" << std::endl;

    if (!ocr_combined.empty()) {
        result.state_flag = "OK";
        result.ocr_text = ocr_combined;
        result.annotated_image = createAnnotatedImage(image, display_infos);
    } else {
        result.state_flag = "NG";
    }

    return result;
}

} // namespace Pipeline
} // namespace JHDeepCore
