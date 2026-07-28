#include "luqian_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

// 把单字符 crop 按比例居中放入 target_w x target_h 画布（pad 0），避免被分类器拉伸变形。
// pm_fx_cls 输入 600x200（宽条带），单字符近正方形，直接送会被 PreprocessImageCommon 的 cv::resize 拉伸失真。
static cv::Mat letterboxToCanvas(const cv::Mat& img, int target_w, int target_h) {
    cv::Mat canvas = cv::Mat::zeros(target_h, target_w, img.type());
    if (img.empty()) return canvas;
    float scale = std::min(static_cast<float>(target_w) / img.cols,
                           static_cast<float>(target_h) / img.rows);
    int new_w = std::max(1, static_cast<int>(std::round(img.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(img.rows * scale)));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    int x = (target_w - new_w) / 2;
    int y = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, new_w, new_h)));
    return canvas;
}


LuqianPipeline::LuqianPipeline(const LuqianServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.det_model, "", dev_id);
    std::cout << "[OK] Det model loaded: " << config_.det_model << std::endl;

    seg_ = std::make_unique<InstanceSegmenter>(config_.seg_model, "", dev_id, "", 0.9f);
    std::cout << "[OK] Seg model loaded: " << config_.seg_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] OCR model loaded: " << config_.ocr_model << std::endl;

    if (!config_.ocr_model2.empty()) {
        // 第二识别模型用独立 ocr_label2 解码；未配置则回退 ocr_label（两模型当前字符表一致，回退安全）
        const std::string& ocr2_label = config_.ocr_label2.empty() ? config_.ocr_label : config_.ocr_label2;
        ocr2_ = std::make_unique<OCRRecognizer>(config_.ocr_model2, ocr2_label, dev_id);
        std::cout << "[OK] OCR2 model loaded: " << config_.ocr_model2
                  << " (label: " << ocr2_label << ")" << std::endl;
    }

    direction_cls_ = std::make_unique<Classifier>(config_.direction_cls_model, "", dev_id);
    std::cout << "[OK] Direction cls model loaded: " << config_.direction_cls_model << std::endl;

    warmup();
}

int LuqianPipeline::classifyDirection(const std::vector<cv::Mat>& char_images,
                                      int* cls_out, float* conf_out)
{
    if (cls_out) *cls_out = -1;
    if (conf_out) *conf_out = 0.f;
    if (char_images.empty()) return 0;
    // 方向分类置信度阈值（对称处理，低置信度一律反向）：
    //   判为翻转(180)且平均置信度 < 阈值 -> 不翻转(0)
    //   判为正向(0)且平均置信度  < 阈值 -> 反向翻转(180)
    constexpr float kFlipConfThreshold = 0.95f;
    // pm_fx_cls 输入 600x200（见 models/luqian/pm_fx_cls.yaml 的 img_scale）
    constexpr int kClsW = 600, kClsH = 200;
    int count_0 = 0, count_180 = 0;
    float conf_0_sum = 0.f, conf_180_sum = 0.f;
    for (auto& img : char_images) {
        if (img.empty()) continue;
        cv::Mat bgr;
        if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        else bgr = img;
        cv::Mat canvas = letterboxToCanvas(bgr, kClsW, kClsH);
        std::vector<cv::Mat> imgs = {canvas};
        std::vector<ClassificationResult> results;
        direction_cls_->process(imgs, results);
        if (!results.empty()) {
            if (results[0].class_id == 0) { count_0++; conf_0_sum += results[0].confidence; }
            else { count_180++; conf_180_sum += results[0].confidence; }
        }
    }
    if (count_180 > count_0) {
        float avg_conf = conf_180_sum / count_180;
        if (cls_out) *cls_out = 180;
        if (conf_out) *conf_out = avg_conf;
        return (avg_conf >= kFlipConfThreshold) ? 180 : 0;
    }
    if (count_0 > count_180) {
        float avg_conf = conf_0_sum / count_0;
        if (cls_out) *cls_out = 0;
        if (conf_out) *conf_out = avg_conf;
        return (avg_conf < kFlipConfThreshold) ? 180 : 0;
    }
    // 票数平局或无有效预测：cls_out 保持 -1，由调用方回退几何法
    return 0;
}

LuqianPipelineResult LuqianPipeline::process(const cv::Mat& image, bool verbose,
                                             const std::string& heat_number)
{
    auto infer_start = std::chrono::high_resolution_clock::now();

    LuqianPipelineResult result;

    // 方向判断：分类器(pm_fx_cls)逐字符投票为主、几何法兜底；heat_number 用于二次识别纠错

    // 单图 OCR 封装：输出识别文本与置信度
    auto runOcr = [&](const cv::Mat& img, std::string& text, float& conf) {
        text.clear();
        conf = 0.f;
        if (img.empty()) return;
        std::vector<cv::Mat> ocr_imgs = {img};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);
        OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];
        if (!ocr_res.boxes.empty()) {
            text = ocr_res.boxes[0].text;
            conf = ocr_res.boxes[0].confidence;
        }
    };

    // 第二 OCR 模型封装（用于二次识别；ocr2_ 为空时直接返回空）
    auto runOcr2 = [&](const cv::Mat& img, std::string& text, float& conf) {
        text.clear();
        conf = 0.f;
        if (img.empty() || !ocr2_) return;
        std::vector<cv::Mat> ocr_imgs = {img};
        std::vector<OCRResult> ocr_results;
        ocr2_->process(ocr_imgs, ocr_results);
        OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];
        if (!ocr_res.boxes.empty()) {
            text = ocr_res.boxes[0].text;
            conf = ocr_res.boxes[0].confidence;
        }
    };

    // ===== Step 1: det 目标检测 =====
    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_res = det_results.empty() ? DetectionResult{} : det_results[0];

    if (verbose) {
        std::cout << "[DEBUG] det detections: " << det_res.num_detections << std::endl;
    }

    result.det_detections = det_res.detections;

    if (det_res.num_detections == 0) {
        result.annotated_image = createAnnotatedImage(image, {}, {});
        return result;
    }

    // 按 x 坐标排序目标检测框
    std::sort(det_res.detections.begin(), det_res.detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.x < b.bbox.x;
        });

    // ===== Step 2: 遍历 det 检测框，裁剪后送实例分割 + 识别 =====
    for (int di = 0; di < det_res.num_detections; di++) {
        auto& det_det = det_res.detections[di];
        cv::Rect target_roi = InferHelper::safeROI(
            det_det.bbox.x, det_det.bbox.y,
            det_det.bbox.width, det_det.bbox.height,
            image.cols, image.rows);
        if (target_roi.area() <= 0) continue;

        LuqianTargetResult target_result;
        target_result.bbox_on_src = target_roi;
        target_result.class_name = det_det.class_name;
        target_result.confidence = det_det.confidence;

        // 裁剪目标区域
        cv::Mat target_img = image(target_roi).clone();
        target_result.target_image = target_img.clone();

        // ===== Step 3: 实例分割 =====
        std::vector<cv::Mat> seg_imgs = {target_img};
        std::vector<InstanceSegmentationResult> seg_results;
        seg_->process(seg_imgs, seg_results);
        InstanceSegmentationResult seg_res = seg_results.empty() ? InstanceSegmentationResult{} : seg_results[0];

        if (verbose) {
            std::cout << "[DEBUG]   target[" << di << "] seg instances: "
                 << seg_res.num_detections << std::endl;
        }

        // 按 y 坐标升序建立原始顺序（自上而下）
        std::vector<int> seg_indices(seg_res.num_detections);
        std::iota(seg_indices.begin(), seg_indices.end(), 0);
        std::sort(seg_indices.begin(), seg_indices.end(), [&](int a, int b) {
            return seg_res.detections[a].bbox.y < seg_res.detections[b].bbox.y;
        });

        // 重排为"最宽 -> 最窄 -> 其他(原 y 顺序)"，再送入 OCR
        if (seg_indices.size() > 1) {
            int widest = 0, narrowest = 0;
            for (int i = 1; i < (int)seg_indices.size(); ++i) {
                int wi = seg_res.detections[seg_indices[i]].bbox.width;
                int ww = seg_res.detections[seg_indices[widest]].bbox.width;
                int wn = seg_res.detections[seg_indices[narrowest]].bbox.width;
                if (wi > ww) widest = i;
                if (wi < wn) narrowest = i;
            }
            std::vector<int> reordered;
            reordered.push_back(seg_indices[widest]);
            if (narrowest != widest)
                reordered.push_back(seg_indices[narrowest]);
            for (int i = 0; i < (int)seg_indices.size(); ++i) {
                if (i != widest && i != narrowest)
                    reordered.push_back(seg_indices[i]);
            }
            seg_indices = std::move(reordered);
        }

        // ===== Step 4: 遍历分割实例，最小外接矩算倾角并水平化，收集字符裁剪框 =====
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
            char_bbox_local = char_bbox_local & cv::Rect(0, 0, target_img.cols, target_img.rows);
            if (char_bbox_local.area() <= 0) continue;

            // 最小外接矩，得到字符倾角。不依赖 min_rect.angle：
            // 不同 OpenCV 版本对 size.width/height 谁大、angle 符号约定不同
            // (旧版 angle 在 [0,90) 且不保证 width>=height；新版归一为 width>=height, angle 在 [-45,45])，
            // 故直接用顶点算长边方向，保证跨版本一致。
            cv::RotatedRect min_rect = cv::minAreaRect(largest);
            cv::Point2f rr_pts[4];
            min_rect.points(rr_pts);
            cv::Point2f e0 = rr_pts[1] - rr_pts[0];
            cv::Point2f e1 = rr_pts[2] - rr_pts[1];
            cv::Point2f long_edge = (cv::norm(e0) > cv::norm(e1)) ? e0 : e1;
            float theta = std::atan2(long_edge.y, long_edge.x) * 180.0f / CV_PI;
            while (theta > 90.0f)  theta -= 180.0f;   // 折到 (-90,90]，横排字符倾角落在 0 附近
            while (theta <= -90.0f) theta += 180.0f;
            float angle = -theta;       // 水平化：旋转抵消倾角

            if (verbose) {
                float raw = min_rect.angle;
                bool vertical = min_rect.size.width < min_rect.size.height;
                std::cout << "[DEBUG]     target[" << di << "] seg[" << si
                     << "] center=(" << min_rect.center.x << "," << min_rect.center.y << ")"
                     << " size=" << min_rect.size.width << "x" << min_rect.size.height
                     << " raw_angle=" << raw
                     << " vertical=" << (vertical ? 1 : 0)
                     << " long_edge_angle=" << theta
                     << " final_angle=" << angle
                     << " area=" << max_area
                     << std::endl;
            }

            // 裁剪字符图像、mask 并转 BGR
            cv::Mat char_img = target_img(char_bbox_local).clone();
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

        // ===== Step 5: 方向判定（分类器投票为主，几何法兜底） =====
        // 字符已各自水平化。先用方向分类器(pm_fx_cls)对每个字符 crop 分类，
        // 多数票 + 置信度阈值定目标朝向；分类器无结论(票数平局/空)时回退几何法
        // （最宽片段是否严格位于其余片段上方）。
        auto geometricAnyFlipped = [&]() -> bool {
            if (char_crops.size() < 2) return false;
            int widest = 0;
            for (int i = 1; i < (int)char_crops.size(); ++i) {
                if (char_crops[i].bbox_local.width > char_crops[widest].bbox_local.width)
                    widest = i;
            }
            int widest_y = char_crops[widest].bbox_local.y;
            for (int i = 0; i < (int)char_crops.size(); ++i) {
                if (i == widest) continue;
                if (widest_y >= char_crops[i].bbox_local.y) return true;  // 非严格在上 -> 倒置
            }
            return false;
        };

        std::vector<cv::Mat> cls_imgs;
        cls_imgs.reserve(char_crops.size());
        for (auto& c : char_crops) cls_imgs.push_back(c.img_bgr);

        int cls_out = -1; float cls_conf = 0.f;
        int cls_dir = classifyDirection(cls_imgs, &cls_out, &cls_conf);

        bool any_flipped;
        if (cls_out >= 0) {
            any_flipped = (cls_dir == 180);
            if (verbose) {
                std::cout << "[DEBUG]   target[" << di << "] direction: " << (any_flipped ? 180 : 0)
                     << " (classifier vote, conf=" << cls_conf << ")" << std::endl;
            }
        } else {
            any_flipped = geometricAnyFlipped();
            if (verbose) {
                std::cout << "[DEBUG]   target[" << di << "] direction: " << (any_flipped ? 180 : 0)
                     << " (geometric fallback)" << std::endl;
            }
        }
        const int target_dir = any_flipped ? 180 : 0;

        // ===== Step 6: 逐片段 OCR（朝向已由 Step 5 确定，整体翻转） =====
        for (auto& c : char_crops) {
            std::string char_text;
            float char_conf = 0.f;
            cv::Mat ocr_in = c.img_bgr;   // 送 OCR 的图（供 after_flip 可视化）

            // 整体翻转 180° 后再 OCR；否则原向 OCR
            if (any_flipped) {
                cv::Mat flipped;
                cv::flip(c.img_bgr, flipped, -1);
                ocr_in = flipped;
                runOcr(flipped, char_text, char_conf);
            } else {
                runOcr(c.img_bgr, char_text, char_conf);
            }

            if (verbose) {
                std::cout << "[DEBUG]     char text=\"" << char_text
                     << "\" flip=" << target_dir
                     << " seg=" << c.confidence
                     << " ocr=" << char_conf << std::endl;
            }

            LuqianCharInfo char_info;
            char_info.bbox_on_target = c.bbox_local;
            char_info.bbox_on_src = cv::Rect(
                target_roi.x + c.bbox_local.x,
                target_roi.y + c.bbox_local.y,
                c.bbox_local.width,
                c.bbox_local.height);
            char_info.mask = c.mask;
            char_info.image_before_flip = c.img_bgr.clone();
            char_info.image_after_flip = ocr_in.clone();
            char_info.ocr_text = char_text;
            char_info.ocr_confidence = char_conf;
            char_info.direction_flag = target_dir;
            target_result.chars.push_back(char_info);
        }

        target_result.direction_flag = target_dir;

        // 片段顺序已由上方"最宽 -> 最窄 -> 其他"重排确定，且该顺序与片段在图像中的
        // 上下位置无关（正/倒置结果一致），故 any_flipped 时不再反向 chars；否则会把最宽
        // 片段翻到末尾，破坏送入 OCR 的读取顺序。

        // ===== Step 7: 二次识别（传入炉号且第二 OCR 模型可用时） =====
        // ocr1_text = 模型1 完整拼接；若与炉号有 1~2 位不符，则用模型2 对全部片段
        // 重识别得 ocr2_text，并逐片段保留使整体与炉号匹配数更多者作为最终结果。
        for (auto& ch : target_result.chars) target_result.ocr1_text += ch.ocr_text;

        if (ocr2_ && !heat_number.empty()) {
            int hlen = (int)heat_number.size();
            int n = (int)std::min(target_result.ocr1_text.size(), (size_t)hlen);
            int matched = 0;
            for (int i = 0; i < n; ++i)
                if (target_result.ocr1_text[i] == heat_number[i]) ++matched;
            int bad = hlen - matched;   // 不符或缺省位数

            if (bad >= 1 && bad <= 2) {
                // 模型2 识别全部片段（完整第二次结果，供对比与选优）
                std::vector<std::string> t2_list(target_result.chars.size());
                std::vector<float> c2_list(target_result.chars.size(), 0.f);
                for (int k = 0; k < (int)target_result.chars.size(); ++k) {
                    runOcr2(target_result.chars[k].image_after_flip, t2_list[k], c2_list[k]);
                    target_result.ocr2_text += t2_list[k];
                }
                // 逐片段比较模型1/模型2，保留使整体匹配更多者
                int cur = matched;
                for (int k = 0; k < (int)target_result.chars.size(); ++k) {
                    if (t2_list[k].empty() || t2_list[k] == target_result.chars[k].ocr_text) continue;
                    std::string old_text = target_result.chars[k].ocr_text;
                    target_result.chars[k].ocr_text = t2_list[k];
                    std::string p;
                    for (auto& ch : target_result.chars) p += ch.ocr_text;
                    int nn = (int)std::min(p.size(), (size_t)hlen);
                    int m2 = 0;
                    for (int i = 0; i < nn; ++i)
                        if (p[i] == heat_number[i]) ++m2;
                    if (m2 > cur) {
                        target_result.chars[k].ocr_confidence = c2_list[k];
                        cur = m2;
                        if (verbose) {
                            std::cout << "[DEBUG]     seg[" << k << "] re-ocr \""
                                 << old_text << "\"->\"" << t2_list[k]
                                 << "\" (matched " << m2 << "/" << hlen << ")" << std::endl;
                        }
                    } else {
                        target_result.chars[k].ocr_text = old_text;
                    }
                }
            }
        }

        // 按当前片段顺序拼接 OCR 文本，并在倒数第三位前插入 '#'
        std::string target_ocr;
        for (auto& ch : target_result.chars) {
            target_ocr += ch.ocr_text;
        }
        if (target_ocr.size() >= 3) {
            target_ocr.insert(target_ocr.size() - 3, "#");
        }
        target_result.ocr_text = target_ocr;

        // 目标置信度 = 各已识别字符(ocr_text 非空)置信度的最小值
        float bmin = 1.0f;
        bool has_rec = false;
        for (auto& ch : target_result.chars) {
            if (!ch.ocr_text.empty()) {
                bmin = std::min(bmin, ch.ocr_confidence);
                has_rec = true;
            }
        }
        target_result.ocr_confidence = has_rec ? bmin : 0.0f;

        if (verbose) {
            std::cout << "[DEBUG]   target[" << di << "] ocr_text=\"" << target_ocr
                 << "\" ocr_conf=" << target_result.ocr_confidence << std::endl;
        }

        result.targets.push_back(target_result);
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) {
        std::cout << "[DEBUG] total inference time: " << infer_ms << " ms" << std::endl;
    }

    result.annotated_image = createAnnotatedImage(
        image, result.det_detections, result.targets);

    return result;
}

cv::Mat LuqianPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<Detection>& det_dets,
    const std::vector<LuqianTargetResult>& targets)
{
    cv::Mat annotated = src_img.clone();

    // 层1: det 目标检测框 (绿色)
    for (int i = 0; i < (int)det_dets.size(); i++) {
        auto& det = det_dets[i];
        cv::rectangle(annotated, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = "det_" + std::to_string(i + 1) + " " + det.class_name
                            + " " + cv::format("%.2f", det.confidence);
        cv::putText(annotated, label,
                    cv::Point(det.bbox.x, det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // 层2+3: 目标检测框 (蓝色) + 实例分割字符 (红色)
    for (int ti = 0; ti < (int)targets.size(); ti++) {
        auto& target = targets[ti];
        cv::rectangle(annotated, target.bbox_on_src, cv::Scalar(255, 100, 0), 2);
        std::string target_label = "target" + std::to_string(ti + 1) + " " + target.class_name
                                    + " " + cv::format("%.2f", target.confidence)
                                    + " dir=" + std::to_string(target.direction_flag);
        cv::putText(annotated, target_label,
                    cv::Point(target.bbox_on_src.x, target.bbox_on_src.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 100, 0), 2);

        for (auto& ch : target.chars) {
            cv::rectangle(annotated, ch.bbox_on_src, cv::Scalar(0, 0, 255), 1);
            // 字符框右上角标注角度分类结果，右下角标注 OCR 置信度
            std::string dir_label = "d" + std::to_string(ch.direction_flag);
            cv::putText(annotated, dir_label,
                        cv::Point(ch.bbox_on_src.x + ch.bbox_on_src.width + 3,
                                  ch.bbox_on_src.y + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
            std::string conf_label = cv::format("%.2f", ch.ocr_confidence);
            cv::putText(annotated, conf_label,
                        cv::Point(ch.bbox_on_src.x + ch.bbox_on_src.width + 3,
                                  ch.bbox_on_src.y + ch.bbox_on_src.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }

    // 层4: 左上角两次识别结果(ocr1/ocr2)与最终结果(final,绿色)，加大字号；超宽时加宽画布
    {
        const double fs = 4.5;     // 字号（加大）
        const int thick = 11;
        const int line_gap = 140;  // 行距
        auto textW = [&](const std::string& s) {
            return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, fs, thick, nullptr).width;
        };
        int need_w = 0;
        for (auto& t : targets) {
            need_w = std::max(need_w, textW("ocr1: " + t.ocr1_text));
            if (!t.ocr2_text.empty())
                need_w = std::max(need_w, textW("ocr2: " + t.ocr2_text));
            need_w = std::max(need_w, textW("final: " + t.ocr_text));
        }
        if (need_w + 40 > annotated.cols) {
            cv::copyMakeBorder(annotated, annotated, 0, 0, 0,
                               need_w + 40 - annotated.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        int text_y = 120;
        for (int ti = 0; ti < (int)targets.size(); ti++) {
            auto& t = targets[ti];
            if (text_y + line_gap > annotated.rows) break;
            cv::putText(annotated, "ocr1: " + t.ocr1_text, cv::Point(20, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 200, 255), thick);   // 黄
            text_y += line_gap;
            if (!t.ocr2_text.empty()) {
                cv::putText(annotated, "ocr2: " + t.ocr2_text, cv::Point(20, text_y),
                            cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(255, 100, 0), thick); // 蓝
                text_y += line_gap;
            }
            cv::putText(annotated, "final: " + t.ocr_text, cv::Point(20, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 255, 0), thick);   // 绿
            text_y += line_gap + 30;
        }
    }

    // ===== 底部可视化：每个目标 = 一张卡片（字符片段），卡片间横向排列、自动换行 =====
    // 参考 zbhc 的可视化方式
    const int margin = 26;
    const int row_h = 300;
    const int char_margin = 8;
    const int gap = 26;
    const int section_gap = 18;
    const int pad_bottom = 26;
    const double strip_font_scale = 1.2;

    // 把多张字符图缩放到统一高度 row_h，并按最大宽度自动折行拼成多行
    // 返回每行 strip（宽度可能不同，高度均为 row_h），供调用方逐行粘贴
    auto buildWrappedStrips = [&](const std::vector<cv::Mat>& imgs, int max_w) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> resized;
        for (auto& img : imgs) {
            if (img.empty()) continue;
            float s = static_cast<float>(row_h) / img.rows;
            cv::Mat r;
            cv::resize(img, r, cv::Size(), s, s, cv::INTER_LINEAR);
            resized.push_back(r);
        }
        std::vector<cv::Mat> lines;
        if (resized.empty()) return lines;
        std::vector<cv::Mat> cur; int cur_w = 0;
        auto flush = [&]() {
            if (cur.empty()) return;
            int total_w = 0;
            for (auto& r : cur) total_w += r.cols + char_margin;
            total_w -= char_margin;
            cv::Mat line = cv::Mat::zeros(row_h, total_w, CV_8UC3);
            int cx = 0;
            for (auto& r : cur) {
                cv::Rect roi(cx, 0, r.cols, r.rows);
                r.copyTo(line(roi));
                cx += r.cols + char_margin;
            }
            lines.push_back(line);
            cur.clear(); cur_w = 0;
        };
        for (auto& r : resized) {
            int add = r.cols + (cur.empty() ? 0 : char_margin);
            if (!cur.empty() && cur_w + add > max_w) flush();
            cur_w += add;
            cur.push_back(r);
        }
        flush();
        return lines;
    };

    const int label_h = 34;            // 每行 strip 上方文字标签高度
    const int line_pitch = row_h + label_h;  // 单行字符（含标签）总高

    // 每个目标打包成一个卡片：字符 strip 多行（自动折行）
    struct Card {
        std::vector<cv::Mat> strip_lines;  // 字符 strip 各行（0°：1+ 行；180°：before 段 + after 段）
        int before_lines = 0;          // 180° 时 before 段行数，用于粘贴时区分标签
        int strip_w = 0;               // strip 区域宽度（取各行最大宽度）
        int strip_h = 0;               // strip 区域高度（含标签）
        int card_w = 0, card_h = 0;    // 整卡尺寸
    };

    int avail_w = annotated.cols - 2 * margin;   // 卡片可用的水平空间
    std::vector<Card> cards;
    cards.reserve(targets.size());

    for (auto& target : targets) {
        Card c;
        int strip_max_w = std::max(1, avail_w);

        std::vector<cv::Mat> imgs_before, imgs_after;
        for (auto& ch : target.chars) {
            imgs_before.push_back(ch.image_before_flip);
            imgs_after.push_back(ch.image_after_flip);
        }

        if (target.direction_flag == 180) {
            auto before = buildWrappedStrips(imgs_before, strip_max_w);
            auto after = buildWrappedStrips(imgs_after, strip_max_w);
            c.before_lines = (int)before.size();
            // before 段
            for (auto& l : before) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)before.size() * line_pitch;
            // after 段（带额外段间距）
            if (!after.empty()) c.strip_h += section_gap;
            for (auto& l : after) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)after.size() * line_pitch;
        } else {
            auto lines = buildWrappedStrips(imgs_after, strip_max_w);
            for (auto& l : lines) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)lines.size() * line_pitch;
        }

        c.card_w = c.strip_lines.empty() ? 0 : c.strip_w;
        c.card_h = c.strip_h;
        cards.push_back(std::move(c));
    }

    // 无有效卡片时不扩展画布
    bool any_content = false;
    for (auto& c : cards) if (c.card_w > 0) { any_content = true; break; }
    if (!any_content) {
        return annotated;
    }

    // 第一遍：流式排版（卡片间横向排列、放不下换行），只算总高度
    int cur_x = margin, cur_y = 0, row_max_h = 0, total_h = 0;
    for (auto& c : cards) {
        if (c.card_w <= 0) continue;
        if (cur_x + c.card_w > annotated.cols - margin && cur_x > margin) {
            cur_y += row_max_h + section_gap;
            cur_x = margin;
            row_max_h = 0;
        }
        cur_x += c.card_w + gap;
        row_max_h = std::max(row_max_h, c.card_h);
        total_h = std::max(total_h, cur_y + row_max_h);
    }
    total_h += pad_bottom;

    // 扩展画布：原图标注置顶，下方追加可视化区域
    int canvas_h = annotated.rows + total_h;
    cv::Mat result = cv::Mat::zeros(canvas_h, annotated.cols, annotated.type());
    annotated.copyTo(result(cv::Rect(0, 0, annotated.cols, annotated.rows)));

    // 带越界裁剪的粘贴（保证绝不出界）
    auto pasteAt = [&](const cv::Mat& img, int x, int y) {
        if (img.empty()) return;
        cv::Rect paste_rect(x, y, img.cols, img.rows);
        cv::Rect clipped = paste_rect & cv::Rect(0, 0, result.cols, result.rows);
        if (clipped.empty()) return;
        cv::Rect src_roi(clipped.x - x, clipped.y - y, clipped.width, clipped.height);
        img(src_roi).copyTo(result(clipped));
    };

    // 第二遍：在确定画布上重新流式排版并粘贴
    cur_x = margin;
    int row_top = annotated.rows;
    row_max_h = 0;
    for (int ti = 0; ti < (int)cards.size(); ti++) {
        auto& c = cards[ti];
        if (c.card_w <= 0) continue;
        std::string tag = "target" + std::to_string(ti + 1);

        // 放不下则换行
        if (cur_x + c.card_w > result.cols - margin && cur_x > margin) {
            row_top += row_max_h + section_gap;
            cur_x = margin;
            row_max_h = 0;
        }

        int card_top = row_top;
        int strip_x = cur_x;

        // 字符 strip 逐行粘贴
        int ly = card_top;
        for (int li = 0; li < (int)c.strip_lines.size(); li++) {
            // 180° 的 after 段首行前补段间距，与 strip_h 计算一致
            if (targets[ti].direction_flag == 180 && li == c.before_lines && c.before_lines > 0) {
                ly += section_gap;
            }
            std::string lbl;
            if (targets[ti].direction_flag == 180) {
                lbl = (li < c.before_lines) ? (tag + " dir=180 Before flip:")
                                            : (tag + " dir=180 After flip:");
            } else {
                lbl = tag + " Char crops:";
            }
            cv::putText(result, lbl, cv::Point(strip_x, ly + label_h - 5),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 3);
            pasteAt(c.strip_lines[li], strip_x, ly + label_h);
            ly += line_pitch;
        }

        cur_x += c.card_w + gap;
        row_max_h = std::max(row_max_h, c.card_h);
    }

    return result;
}

void LuqianPipeline::warmup()
{
    std::cout << "[INFO] Warming up models..." << std::endl;
    std::vector<cv::Mat> dummy_imgs = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det_res;
    det_->process(dummy_imgs, det_res);
    std::cout << "[OK] Det warmed up" << std::endl;

    std::vector<InstanceSegmentationResult> seg_res;
    seg_->process(dummy_imgs, seg_res);
    std::cout << "[OK] Seg warmed up" << std::endl;

    std::vector<OCRResult> ocr_res;
    ocr_->process(dummy_imgs, ocr_res);
    std::cout << "[OK] OCR warmed up" << std::endl;

    if (ocr2_) {
        std::vector<OCRResult> ocr2_res;
        ocr2_->process(dummy_imgs, ocr2_res);
        std::cout << "[OK] OCR2 warmed up" << std::endl;
    }

    std::vector<ClassificationResult> cls_res;
    direction_cls_->process(dummy_imgs, cls_res);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

} // namespace Pipeline
} // namespace JHDeepCore
