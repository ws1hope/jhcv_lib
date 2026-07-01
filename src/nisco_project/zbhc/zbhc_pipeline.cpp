#include "zbhc_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

ZbhcPipeline::ZbhcPipeline(const ZbhcServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det1_ = std::make_unique<Detector>(config_.det1_model, "", dev_id);
    std::cout << "[OK] Det1 model loaded: " << config_.det1_model << std::endl;

    det2_ = std::make_unique<Detector>(config_.det2_model, "", dev_id);
    std::cout << "[OK] Det2 model loaded: " << config_.det2_model << std::endl;

    seg_ = std::make_unique<InstanceSegmenter>(config_.seg_model, "", dev_id);
    std::cout << "[OK] Seg model loaded: " << config_.seg_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] OCR model loaded: " << config_.ocr_model << std::endl;

    direction_cls_ = std::make_unique<Classifier>(config_.direction_cls_model, "", dev_id);
    std::cout << "[OK] Direction cls model loaded: " << config_.direction_cls_model << std::endl;

    warmup();
}

ZbhcPipelineResult ZbhcPipeline::process(const cv::Mat& image, bool verbose)
{
    auto infer_start = std::chrono::high_resolution_clock::now();

    ZbhcPipelineResult result;

    // ===== Step 1: det1 整体范围检测 =====
    std::vector<cv::Mat> det1_imgs = {image};
    std::vector<DetectionResult> det1_results;
    det1_->process(det1_imgs, det1_results);
    DetectionResult det1_res = det1_results.empty() ? DetectionResult{} : det1_results[0];

    if (verbose) {
        std::cout << "[DEBUG] det1 detections: " << det1_res.num_detections << std::endl;
    }

    result.det1_detections = det1_res.detections;

    if (det1_res.num_detections == 0) {
        result.annotated_image = createAnnotatedImage(image, {}, {});
        return result;
    }

    // ===== Step 2: 遍历 det1 检测框，裁剪后送 det2 =====
    for (int di = 0; di < det1_res.num_detections; di++) {
        auto& det1_det = det1_res.detections[di];
        cv::Rect det1_roi = InferHelper::safeROI(
            det1_det.bbox.x, det1_det.bbox.y,
            det1_det.bbox.width, det1_det.bbox.height,
            image.cols, image.rows);
        if (det1_roi.area() <= 0) continue;

        cv::Mat roi_img = image(det1_roi).clone();

        // ===== Step 3: det2 坯料检测 =====
        std::vector<cv::Mat> det2_imgs = {roi_img};
        std::vector<DetectionResult> det2_results;
        det2_->process(det2_imgs, det2_results);
        DetectionResult det2_res = det2_results.empty() ? DetectionResult{} : det2_results[0];

        if (verbose) {
            std::cout << "[DEBUG] det1[" << di << "] -> det2 detections: "
                 << det2_res.num_detections << std::endl;
        }

        // 按 x 坐标排序坯料检测框
        std::sort(det2_res.detections.begin(), det2_res.detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.bbox.x < b.bbox.x;
            });

        // ===== Step 4: 遍历每个坯料，送实例分割 + OCR =====
        for (int bi = 0; bi < det2_res.num_detections; bi++) {
            auto& billet_det = det2_res.detections[bi];
            cv::Rect billet_roi_local = InferHelper::safeROI(
                billet_det.bbox.x, billet_det.bbox.y,
                billet_det.bbox.width, billet_det.bbox.height,
                roi_img.cols, roi_img.rows);
            if (billet_roi_local.area() <= 0) continue;

            // 映射回原图坐标
            cv::Rect billet_roi_on_src(
                det1_roi.x + billet_roi_local.x,
                det1_roi.y + billet_roi_local.y,
                billet_roi_local.width,
                billet_roi_local.height);
            billet_roi_on_src = InferHelper::safeROI(
                billet_roi_on_src.x, billet_roi_on_src.y,
                billet_roi_on_src.width, billet_roi_on_src.height,
                image.cols, image.rows);
            if (billet_roi_on_src.area() <= 0) continue;

            BilletResult billet_result;
            billet_result.bbox_on_src = billet_roi_on_src;
            billet_result.class_name = billet_det.class_name;
            billet_result.confidence = billet_det.confidence;

            // 裁剪坯料区域
            cv::Mat billet_img = image(billet_roi_on_src).clone();
            billet_result.billet_image = billet_img.clone();

            // ===== Step 5: 实例分割 =====
            std::vector<cv::Mat> seg_imgs = {billet_img};
            std::vector<InstanceSegmentationResult> seg_results;
            seg_->process(seg_imgs, seg_results);
            InstanceSegmentationResult seg_res = seg_results.empty() ? InstanceSegmentationResult{} : seg_results[0];

            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] seg instances: "
                     << seg_res.num_detections << std::endl;
            }

            // 按 x 坐标排序分割实例
            std::vector<int> seg_indices(seg_res.num_detections);
            std::iota(seg_indices.begin(), seg_indices.end(), 0);
            std::sort(seg_indices.begin(), seg_indices.end(), [&](int a, int b) {
                return seg_res.detections[a].bbox.x < seg_res.detections[b].bbox.x;
            });

            // ===== Step 6: 遍历分割实例，最小外接矩算倾角并水平化，收集字符裁剪框 =====
            struct CharCrop {
                cv::Rect bbox_local;
                cv::Mat mask;
                cv::Mat img_bgr;
                float confidence;
            };
            std::vector<CharCrop> char_crops;

            for (int si : seg_indices) {
                auto& char_det = seg_res.detections[si];
                cv::Mat char_mask = seg_res.masks[si];

                // 从 mask 提取轮廓，取最大轮廓计算外接矩形与倾角
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(char_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (contours.empty()) continue;

                int largest_idx = 0;
                double max_area = 0.0;
                for (int k = 0; k < (int)contours.size(); k++) {
                    double a = cv::contourArea(contours[k]);
                    if (a > max_area) { max_area = a; largest_idx = k; }
                }
                const std::vector<cv::Point>& largest = contours[largest_idx];

                cv::Rect char_bbox_local = cv::boundingRect(largest);
                char_bbox_local = char_bbox_local & cv::Rect(0, 0, billet_img.cols, billet_img.rows);
                if (char_bbox_local.area() <= 0) continue;

                // 最小外接矩，得到字符倾角（minAreaRect.angle ∈ [-90,0)）
                cv::RotatedRect min_rect = cv::minAreaRect(largest);
                float angle = min_rect.angle;
                if (min_rect.size.width < min_rect.size.height) {
                    angle += 90.0f;   // 竖立矩归一到水平方向
                }
                angle = -angle;       // 水平化：旋转抵消倾角

                // 裁剪字符图像、mask 并转 BGR
                cv::Mat char_img = billet_img(char_bbox_local).clone();
                cv::Mat char_mask_crop = char_mask(char_bbox_local).clone();
                if (char_img.empty()) continue;
                cv::Mat char_img_bgr;
                if (char_img.channels() == 1) {
                    cv::cvtColor(char_img, char_img_bgr, cv::COLOR_GRAY2BGR);
                } else {
                    char_img_bgr = char_img;
                }

                // 水平化：加边距后绕字符中心旋转，避免旋转时裁切字符
                // pad 取 W+H（>= 对角线，任意角度都不裁切），旋转后再紧裁回去
                int W = char_bbox_local.width, H = char_bbox_local.height;
                int pad = W + H + 2;
                cv::Mat padded_img, padded_mask;
                cv::copyMakeBorder(char_img_bgr, padded_img, pad, pad, pad, pad,
                                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                cv::copyMakeBorder(char_mask_crop, padded_mask, pad, pad, pad, pad,
                                   cv::BORDER_CONSTANT, cv::Scalar(0));
                cv::Point2f rot_center(
                    min_rect.center.x - char_bbox_local.x + pad,
                    min_rect.center.y - char_bbox_local.y + pad);
                cv::Mat rotM = cv::getRotationMatrix2D(rot_center, angle, 1.0);
                cv::Mat leveled_img, leveled_mask;
                cv::warpAffine(padded_img, leveled_img, rotM, padded_img.size(),
                               cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                cv::warpAffine(padded_mask, leveled_mask, rotM, padded_mask.size(),
                               cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

                // 旋转后重新紧裁：用旋转后的 mask 取轴对齐外接框去掉 padding，
                // 否则 padding 会被方向分类/OCR/可视化当作字符的一部分，导致字符被等比缩小
                cv::Rect tight = cv::boundingRect(leveled_mask);
                int m = 2;
                tight.x = std::max(0, tight.x - m);
                tight.y = std::max(0, tight.y - m);
                tight.width = std::min(leveled_img.cols - tight.x, tight.width + 2 * m);
                tight.height = std::min(leveled_img.rows - tight.y, tight.height + 2 * m);
                if (tight.area() <= 0) continue;
                cv::Mat leveled = leveled_img(tight).clone();

                CharCrop crop;
                crop.bbox_local = char_bbox_local;
                crop.mask = char_mask;
                crop.img_bgr = leveled;   // 水平化并紧裁后的字符（供方向分类与 OCR）
                crop.confidence = char_det.confidence;
                char_crops.push_back(std::move(crop));
            }

            // ===== Step 7: 方向分类(0/180)，坯料内投票决定角度 =====
            std::vector<cv::Mat> crop_imgs;
            for (auto& c : char_crops) crop_imgs.push_back(c.img_bgr);
            int dir_flag = classifyDirection(crop_imgs);
            billet_result.direction_flag = dir_flag;

            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] direction: " << dir_flag << std::endl;
            }

            // ===== Step 8: 方向矫正(180°则翻转)后送 OCR 识别 =====
            for (auto& c : char_crops) {
                cv::Mat ocr_in = c.img_bgr;
                if (dir_flag == 180 && !ocr_in.empty()) {
                    cv::Mat flipped;
                    cv::flip(ocr_in, flipped, -1);
                    ocr_in = flipped;
                }

                std::vector<cv::Mat> ocr_imgs = {ocr_in};
                std::vector<OCRResult> ocr_results;
                ocr_->process(ocr_imgs, ocr_results);
                OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];

                std::string char_text;
                if (!ocr_res.boxes.empty()) {
                    char_text = ocr_res.boxes[0].text;
                }

                if (verbose) {
                    std::cout << "[DEBUG]     char text=\"" << char_text
                         << "\" conf=" << c.confidence << std::endl;
                }

                BilletCharInfo char_info;
                char_info.bbox_on_billet = c.bbox_local;
                char_info.bbox_on_src = cv::Rect(
                    billet_roi_on_src.x + c.bbox_local.x,
                    billet_roi_on_src.y + c.bbox_local.y,
                    c.bbox_local.width,
                    c.bbox_local.height);
                char_info.mask = c.mask;
                char_info.image_before_flip = c.img_bgr.clone();
                char_info.image_after_flip = ocr_in.clone();
                char_info.ocr_text = char_text;
                billet_result.chars.push_back(char_info);
            }

            // 拼接坯料内所有字符的 OCR 结果
            std::string billet_ocr;
            for (auto& ch : billet_result.chars) {
                billet_ocr += ch.ocr_text;
            }
            billet_result.ocr_text = billet_ocr;

            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] ocr_text=\"" << billet_ocr << "\"" << std::endl;
            }

            result.billets.push_back(billet_result);
        }
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) {
        std::cout << "[DEBUG] total inference time: " << infer_ms << " ms" << std::endl;
    }

    result.annotated_image = createAnnotatedImage(
        image, result.det1_detections, result.billets);

    return result;
}

cv::Mat ZbhcPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<Detection>& det1_dets,
    const std::vector<BilletResult>& billets)
{
    cv::Mat annotated = src_img.clone();

    // 层1: det1 全局检测框 (绿色)
    for (int i = 0; i < (int)det1_dets.size(); i++) {
        auto& det = det1_dets[i];
        cv::rectangle(annotated, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = "det1_" + std::to_string(i + 1) + " " + det.class_name
                            + " " + cv::format("%.2f", det.confidence);
        cv::putText(annotated, label,
                    cv::Point(det.bbox.x, det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // 层2+3: det2 坯料检测框 (蓝色) + 实例分割字符 (红色)
    for (int bi = 0; bi < (int)billets.size(); bi++) {
        auto& billet = billets[bi];
        cv::rectangle(annotated, billet.bbox_on_src, cv::Scalar(255, 100, 0), 2);
        std::string billet_label = "billet" + std::to_string(bi + 1) + " " + billet.class_name
                                    + " " + cv::format("%.2f", billet.confidence)
                                    + " dir=" + std::to_string(billet.direction_flag);
        cv::putText(annotated, billet_label,
                    cv::Point(billet.bbox_on_src.x, billet.bbox_on_src.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 100, 0), 2);

        for (auto& ch : billet.chars) {
            cv::rectangle(annotated, ch.bbox_on_src, cv::Scalar(0, 0, 255), 1);
        }
    }

    // 层4: 左上角 OCR 结果文本 (红色)
    int text_y = 120;
    for (int bi = 0; bi < (int)billets.size(); bi++) {
        if (text_y + 20 > annotated.rows) break;
        std::string text = "billet" + std::to_string(bi + 1) + ": " + billets[bi].ocr_text;
        cv::putText(annotated, text, cv::Point(20, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 3.6, cv::Scalar(0, 0, 255), 9);
        text_y += 150;
    }

    // ===== 底部可视化：每个坯料一行 = 坯料缩略图 + 方向矫正前/后字符片段 =====
    // 参考 gx_jingzheng tiebiao branch 的可视化方式
    const int margin = 15;
    const int row_h = 60;
    const int thumb_height = 200;
    const int char_margin = 4;
    const int gap = 15;
    const int section_gap = 10;
    const int pad_bottom = 15;
    const double strip_font_scale = 0.7;

    // 计算底部区域高度
    int bottom_area_h = 0;
    for (auto& billet : billets) {
        int num_char_rows = (billet.direction_flag == 180) ? 2 : 1;
        int strip_total_h = num_char_rows * (row_h + section_gap + 20);
        int row_total = std::max(thumb_height, strip_total_h);
        bottom_area_h += row_total + section_gap;
    }

    // 无坯料时不扩展画布
    if (bottom_area_h == 0) {
        return annotated;
    }

    // 扩展画布：原图标注置顶，下方追加可视化区域
    int canvas_h = annotated.rows + bottom_area_h + pad_bottom;
    cv::Mat result = cv::Mat::zeros(canvas_h, annotated.cols, annotated.type());
    annotated.copyTo(result(cv::Rect(0, 0, annotated.cols, annotated.rows)));

    // 把多张字符图水平拼接成统一高度 row_h 的长条
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

    // 带越界裁剪的粘贴
    auto pasteAt = [&](const cv::Mat& img, int x, int y) {
        if (img.empty()) return;
        cv::Rect paste_rect(x, y, img.cols, img.rows);
        cv::Rect clipped = paste_rect & cv::Rect(0, 0, result.cols, result.rows);
        if (clipped.empty()) return;
        cv::Rect src_roi(clipped.x - x, clipped.y - y, clipped.width, clipped.height);
        img(src_roi).copyTo(result(clipped));
    };

    int bottom_y = annotated.rows;

    for (int bi = 0; bi < (int)billets.size(); bi++) {
        auto& billet = billets[bi];
        std::string tag = "billet" + std::to_string(bi + 1);

        // 坯料缩略图（左侧）—— zbhc 的检测对象是坯料，无"圆标"概念，以坯料 crop 作为展示对象
        cv::Mat thumb;
        if (!billet.billet_image.empty()) {
            float thumb_scale = static_cast<float>(thumb_height) / billet.billet_image.rows;
            cv::resize(billet.billet_image, thumb, cv::Size(), thumb_scale, thumb_scale, cv::INTER_LINEAR);
        }
        int thumb_w = thumb.empty() ? 0 : thumb.cols;
        int strip_x = margin + thumb_w + gap;

        // 收集字符片段（矫正前 / 矫正后）
        std::vector<cv::Mat> imgs_before, imgs_after;
        for (auto& ch : billet.chars) {
            imgs_before.push_back(ch.image_before_flip);
            imgs_after.push_back(ch.image_after_flip);
        }

        int cur_y = bottom_y;
        if (!thumb.empty()) {
            pasteAt(thumb, margin, cur_y);
        }

        if (billet.direction_flag == 180) {
            // 180°：展示矫正前 + 矫正后两行，便于对比
            cv::Mat strip_before = buildCharStrip(imgs_before);
            cv::Mat strip_after = buildCharStrip(imgs_after);

            int strip_y = cur_y + 20;
            cv::putText(result, tag + " dir=180 Before flip:", cv::Point(strip_x, strip_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_before, strip_x, strip_y + 10);

            int strip2_y = strip_y + row_h + section_gap + 10;
            cv::putText(result, tag + " dir=180 After flip:", cv::Point(strip_x, strip2_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_after, strip_x, strip2_y + 10);

            bottom_y = cur_y + thumb_height + section_gap;
        } else {
            // 0°：仅展示字符片段（矫正前后一致）
            cv::Mat strip_crops = buildCharStrip(imgs_after);
            int strip_y = cur_y + (thumb_height - row_h) / 2;
            cv::putText(result, tag + " Char crops:", cv::Point(strip_x, strip_y),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(strip_crops, strip_x, strip_y + 10);
            bottom_y = cur_y + thumb_height + section_gap;
        }
    }

    return result;
}

void ZbhcPipeline::warmup()
{
    std::cout << "[INFO] Warming up models..." << std::endl;
    std::vector<cv::Mat> dummy_imgs = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det1_res;
    det1_->process(dummy_imgs, det1_res);
    std::cout << "[OK] Det1 warmed up" << std::endl;

    std::vector<DetectionResult> det2_res;
    det2_->process(dummy_imgs, det2_res);
    std::cout << "[OK] Det2 warmed up" << std::endl;

    std::vector<InstanceSegmentationResult> seg_res;
    seg_->process(dummy_imgs, seg_res);
    std::cout << "[OK] Seg warmed up" << std::endl;

    std::vector<OCRResult> ocr_res;
    ocr_->process(dummy_imgs, ocr_res);
    std::cout << "[OK] OCR warmed up" << std::endl;

    std::vector<ClassificationResult> cls_res;
    direction_cls_->process(dummy_imgs, cls_res);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

int ZbhcPipeline::classifyDirection(const std::vector<cv::Mat>& char_images)
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

} // namespace Pipeline
} // namespace JHDeepCore
