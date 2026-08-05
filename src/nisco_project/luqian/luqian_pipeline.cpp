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

// 把一次 lastBatchTiming() 的结果累加进 per-model 累加器
static void accumulateTiming(InferenceTiming& acc, const InferenceTiming& t)
{
    acc.count += t.count;
    acc.preprocess_ms += t.preprocess_ms;
    acc.tensor_ms += t.tensor_ms;
    acc.run_ms += t.run_ms;
    acc.h2d_ms += t.h2d_ms;
    acc.d2h_ms += t.d2h_ms;
    acc.gpu_total_ms += t.gpu_total_ms;
    acc.wall_ms += t.wall_ms;
    acc.h2d_split = acc.h2d_split || t.h2d_split;
    acc.gpu_timing_valid = acc.gpu_timing_valid || t.gpu_timing_valid;
    if (acc.device.empty() && !t.device.empty()) acc.device = t.device;
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
        accumulateTiming(t_cls_, direction_cls_->lastBatchTiming());
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

void LuqianPipeline::fillTiming(LuqianPipelineResult& r, double total_ms)
{
    r.timing.det = t_det_;
    r.timing.seg = t_seg_;
    r.timing.cls = t_cls_;
    r.timing.ocr = t_ocr_;
    r.timing.ocr2 = t_ocr2_;
    r.timing.total_ms = total_ms;
    // device 取首个有值的模型（实际执行设备，cuda 下可能因 EP 失败降级为 cpu）
    if (!t_det_.device.empty()) r.timing.device = t_det_.device;
    else if (!t_seg_.device.empty()) r.timing.device = t_seg_.device;
    else if (!t_cls_.device.empty()) r.timing.device = t_cls_.device;
    else if (!t_ocr_.device.empty()) r.timing.device = t_ocr_.device;
    else if (!t_ocr2_.device.empty()) r.timing.device = t_ocr2_.device;
    else r.timing.device = config_.device;
}

LuqianPipelineResult LuqianPipeline::process(const cv::Mat& image, bool verbose,
                                             const std::string& heat_number)
{
    auto infer_start = std::chrono::steady_clock::now();

    LuqianPipelineResult result;

    // 复位各模型耗时累加器
    t_det_ = t_seg_ = t_cls_ = t_ocr_ = t_ocr2_ = InferenceTiming{};

    // 方向判断：分类器(pm_fx_cls)逐字符投票为主、几何法兜底；heat_number 用于二次识别纠错

    // 单图 OCR 封装：输出识别文本与置信度
    auto runOcr = [&](const cv::Mat& img, std::string& text, float& conf) {
        text.clear();
        conf = 0.f;
        if (img.empty()) return;
        std::vector<cv::Mat> ocr_imgs = {img};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);
        accumulateTiming(t_ocr_, ocr_->lastBatchTiming());
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
        accumulateTiming(t_ocr2_, ocr2_->lastBatchTiming());
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
    accumulateTiming(t_det_, det_->lastBatchTiming());
    DetectionResult det_res = det_results.empty() ? DetectionResult{} : det_results[0];

    if (verbose) {
        std::cout << "[DEBUG] det detections: " << det_res.num_detections << std::endl;
    }

    result.det_detections = det_res.detections;

    if (det_res.num_detections == 0) {
        auto infer_end = std::chrono::steady_clock::now();
        auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
        fillTiming(result, static_cast<double>(infer_ms));
        result.annotated_image = createAnnotatedImage(image, {});
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
        accumulateTiming(t_seg_, seg_->lastBatchTiming());
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

        // ===== Step 5: 方向判定（几何法为主、分类器兜底） =====
        // 字符已各自水平化，但片段间相对位置不变。最宽(最长)字符片段若位于
        // 其余片段上方(y 更小)，则整体为正向；否则为倒置，整体翻转 180°。
        // 触发：最宽与最窄片段宽度差 >= 300（最宽片段足够突出，几何信号可靠）时
        // 用几何法；否则用方向分类器(pm_fx_cls)逐字符投票判向，分类器无结论时
        // 回退几何法。
        auto geometricAnyFlipped = [&]() -> bool {
            if (char_crops.size() < 2) return false;
            int widest = 0;
            for (int i = 1; i < (int)char_crops.size(); ++i) {
                if (char_crops[i].bbox_local.width > char_crops[widest].bbox_local.width)
                    widest = i;
            }
            int widest_y = char_crops[widest].bbox_local.y;
            bool upright = true;  // 最宽片段是否严格位于所有其余片段上方
            for (int i = 0; i < (int)char_crops.size(); ++i) {
                if (i == widest) continue;
                if (widest_y >= char_crops[i].bbox_local.y) { upright = false; break; }
            }
            return !upright;
        };

        // 最宽与最窄片段宽度差 >= 300 时几何信号可靠，优先用几何法
        bool use_geometric = false;
        if (char_crops.size() >= 2) {
            int widest = 0, narrowest = 0;
            for (int i = 1; i < (int)char_crops.size(); ++i) {
                if (char_crops[i].bbox_local.width > char_crops[widest].bbox_local.width) widest = i;
                if (char_crops[i].bbox_local.width < char_crops[narrowest].bbox_local.width) narrowest = i;
            }
            use_geometric = (char_crops[widest].bbox_local.width
                             - char_crops[narrowest].bbox_local.width) >= 300;
        }

        bool any_flipped;
        if (use_geometric) {
            any_flipped = geometricAnyFlipped();
            if (verbose) {
                std::cout << "[DEBUG]   target[" << di << "] direction: " << (any_flipped ? 180 : 0)
                     << " (geometric, width gap>=300)" << std::endl;
            }
        } else {
            std::vector<cv::Mat> cls_imgs;
            cls_imgs.reserve(char_crops.size());
            for (auto& c : char_crops) cls_imgs.push_back(c.img_bgr);

            int cls_out = -1; float cls_conf = 0.f;
            int cls_dir = classifyDirection(cls_imgs, &cls_out, &cls_conf);
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
                         << " (geometric fallback, classifier undecided)" << std::endl;
                }
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

        // OCR 后按内容对齐炉号开头：把与输入炉号逐字符匹配最多（无炉号则取首两位
        // 数字 >= 26）的片段稳定置首，其余保持原相对顺序。基于 model1 的 ocr_text。
        if (target_result.chars.size() >= 2) {
            int first_idx = -1;
            if (!heat_number.empty()) {
                int best_match = -1;
                for (int k = 0; k < (int)target_result.chars.size(); ++k) {
                    const std::string& t = target_result.chars[k].ocr_text;
                    size_t n = std::min(t.size(), heat_number.size());
                    int m = 0;
                    for (size_t i = 0; i < n; ++i) if (t[i] == heat_number[i]) ++m;
                    if (m > best_match) { best_match = m; first_idx = k; }
                }
            } else {
                for (int k = 0; k < (int)target_result.chars.size(); ++k) {
                    const std::string& t = target_result.chars[k].ocr_text;
                    if (t.size() >= 2 && t[0] >= '0' && t[0] <= '9' && t[1] >= '0' && t[1] <= '9') {
                        if ((t[0] - '0') * 10 + (t[1] - '0') >= 26) { first_idx = k; break; }
                    }
                }
            }
            if (first_idx > 0) {
                std::rotate(target_result.chars.begin(),
                            target_result.chars.begin() + first_idx,
                            target_result.chars.begin() + first_idx + 1);
            }
        }

        // ===== Step 7: 二次识别（传入炉号且第二 OCR 模型可用时） =====
        // ocr1_text = 模型1 完整拼接；若与炉号有任意位不符(bad>=1)，则用模型2 对全部片段
        // 重识别得 ocr2_text，并逐片段保留使整体与炉号匹配数更多者作为最终结果。
        for (auto& ch : target_result.chars) target_result.ocr1_text += ch.ocr_text;

        if (ocr2_ && !heat_number.empty()) {
            int hlen = (int)heat_number.size();
            int n = (int)std::min(target_result.ocr1_text.size(), (size_t)hlen);
            int matched = 0;
            for (int i = 0; i < n; ++i)
                if (target_result.ocr1_text[i] == heat_number[i]) ++matched;
            int bad = hlen - matched;   // 不符或缺省位数

            if (verbose) {
                std::cout << "[DEBUG]     target[" << di << "] ocr2 gate:"
                     << " heat=\"" << heat_number << "\""
                     << " ocr1=\"" << target_result.ocr1_text << "\""
                     << " hlen=" << hlen
                     << " matched=" << matched
                     << " bad=" << bad
                     << " -> " << (bad >= 1 ? "retry" : "skip")
                     << std::endl;
            }

            if (bad >= 1) {
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
        } else if (verbose) {
            std::cout << "[DEBUG]     target[" << di << "] ocr2 gate: skip ("
                 << (!ocr2_ ? "no ocr2 model" : "empty heat_number") << ")"
                 << std::endl;
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

        // ===== Step 8: 炉号纠错兜底 =====
        // 已传入炉号且两次识别(ocr1/ocr2)均有结果时：若两者都未与炉号完全一致，
        // 但其中之一与炉号错位 ≤3（识别已接近正确），则改用用户输入的权威炉号
        // 作为最终结果。错位口径与 Step 7 的 bad 一致：仅按炉号长度 hlen 对齐前缀
        // 计数(hlen - matched)，忽略识别串超出 hlen 的尾部(如 '#' 后的附加后缀)，
        // 故不会把附加码误计为错位。仅替换热号部分，保留原识别的后缀。
        if (!heat_number.empty() && !target_result.ocr2_text.empty()) {
            auto heatErrors = [](const std::string& ocr, const std::string& heat) -> int {
                int hlen = (int)heat.size();
                int n = (int)std::min(ocr.size(), (size_t)hlen);
                int matched = 0;
                for (int i = 0; i < n; ++i) if (ocr[i] == heat[i]) ++matched;
                return hlen - matched;
            };
            int e1 = heatErrors(target_result.ocr1_text, heat_number);
            int e2 = heatErrors(target_result.ocr2_text, heat_number);
            // e==0 表示该次识别的热号部分与炉号完全一致；两者都 >0 才算"都没比对上"
            if (e1 > 0 && e2 > 0 && std::min(e1, e2) <= 3) {
                std::string heat_final = heat_number;
                size_t hp = target_result.ocr_text.find('#');
                if (hp != std::string::npos) {
                    // 保留原识别结果 '#' 后的尾部后缀，仅替换热号部分
                    heat_final += "#";
                    heat_final += target_result.ocr_text.substr(hp + 1);
                } else if (heat_final.size() >= 3) {
                    heat_final.insert(heat_final.size() - 3, "#");
                }
                target_result.ocr_text = heat_final;
                if (verbose) {
                    std::cout << "[DEBUG]     target[" << di << "] heat override:"
                         << " ocr1_err=" << e1 << " ocr2_err=" << e2
                         << " -> \"" << heat_final << "\"" << std::endl;
                }
            }
        }

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
            std::cout << "[DEBUG]   target[" << di << "] ocr_text=\"" << target_result.ocr_text
                 << "\" ocr_conf=" << target_result.ocr_confidence << std::endl;
        }

        result.targets.push_back(target_result);
    }

    auto infer_end = std::chrono::steady_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    fillTiming(result, static_cast<double>(infer_ms));
    if (verbose) {
        std::cout << "[DEBUG] luqian timing summary (device=" << result.timing.device << "):" << std::endl;
        auto printRow = [&](const char* name, const InferenceTiming& t) {
            std::cout << "[DEBUG]   " << name << " : n=" << t.count
                      << " prep=" << t.preprocess_ms << "ms";
            if (t.h2d_split) {
                std::cout << " h2d=" << t.h2d_ms << "ms"
                          << " d2h=" << t.d2h_ms << "ms"
                          << " infer=" << t.run_ms << "ms";
                if (t.gpu_timing_valid) {
                    std::cout << " gpu=" << t.gpu_total_ms << "ms"
                              << " wall=" << t.wall_ms << "ms";
                }
            } else {
                std::cout << " ten=" << t.tensor_ms << "ms"
                          << " run=" << t.run_ms << "ms (incl. H2D)";
            }
            std::cout << std::endl;
        };
        printRow("det", result.timing.det);
        printRow("seg", result.timing.seg);
        printRow("cls", result.timing.cls);
        printRow("ocr", result.timing.ocr);
        printRow("ocr2", result.timing.ocr2);
        std::cout << "[DEBUG]   total: " << result.timing.total_ms << " ms" << std::endl;
    }

    result.annotated_image = createAnnotatedImage(
        image, result.targets);

    return result;
}

cv::Mat LuqianPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<LuqianTargetResult>& targets)
{
    // 底图裁到第一个目标框（targets[0].bbox_on_src = 最左 det 框）内；无目标时回退整图。
    // 后续绿/蓝/红框及标注均按裁剪原点偏移对齐，且只绘制第一个目标。
    cv::Rect crop_rect(0, 0, src_img.cols, src_img.rows);
    const bool focus_first = !targets.empty();
    if (focus_first) {
        cv::Rect r = targets[0].bbox_on_src & cv::Rect(0, 0, src_img.cols, src_img.rows);
        if (r.area() > 0) crop_rect = r;
    }
    cv::Mat annotated = src_img(crop_rect).clone();
    const int ox = crop_rect.x, oy = crop_rect.y;   // 裁剪原点：整图坐标减去它落到裁剪图
    auto off = [&](cv::Rect r) {
        return cv::Rect(r.x - ox, r.y - oy, r.width, r.height);
    };

    if (focus_first) {
        const auto& t0 = targets[0];

        // 层1: det 目标检测框 (绿色) - 第一个目标的 det 框
        {
            cv::Rect dr = off(t0.bbox_on_src);
            cv::rectangle(annotated, dr, cv::Scalar(0, 255, 0), 2);
            std::string label = "det_1 " + t0.class_name
                                + " " + cv::format("%.2f", t0.confidence);
            cv::putText(annotated, label,
                        cv::Point(dr.x, dr.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // 层2+3: 目标检测框 (蓝色) + 实例分割字符 (红色)
        {
            cv::Rect tr = off(t0.bbox_on_src);
            cv::rectangle(annotated, tr, cv::Scalar(255, 100, 0), 2);
            std::string target_label = "target1 " + t0.class_name
                                        + " " + cv::format("%.2f", t0.confidence)
                                        + " dir=" + std::to_string(t0.direction_flag);
            cv::putText(annotated, target_label,
                        cv::Point(tr.x, tr.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 100, 0), 2);

            for (auto& ch : t0.chars) {
                cv::Rect cr = off(ch.bbox_on_src);
                cv::rectangle(annotated, cr, cv::Scalar(0, 0, 255), 1);
                // 字符框右上角标注角度分类结果，右下角标注 OCR 置信度
                std::string dir_label = "d" + std::to_string(ch.direction_flag);
                cv::putText(annotated, dir_label,
                            cv::Point(cr.x + cr.width + 3, cr.y + 12),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
                std::string conf_label = cv::format("%.2f", ch.ocr_confidence);
                cv::putText(annotated, conf_label,
                            cv::Point(cr.x + cr.width + 3, cr.y + cr.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    // 层4: 左上角最终结果(final,绿色)，加大字号；超宽时加宽画布
    // ocr1/ocr2 不再绘于图上，改写入日志（见 luqian_service.cpp handleRequest 的 [ocr] 行）
    if (focus_first) {
        const auto& t = targets[0];
        const double fs = 4.5;     // 字号（加大）
        const int thick = 11;
        const int line_gap = 140;  // 行距
        auto textW = [&](const std::string& s) {
            return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, fs, thick, nullptr).width;
        };
        int need_w = textW(t.ocr_text);
        if (need_w + 40 > annotated.cols) {
            cv::copyMakeBorder(annotated, annotated, 0, 0, 0,
                               need_w + 40 - annotated.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        int text_y = 120;
        if (text_y + line_gap <= annotated.rows) {
            cv::putText(annotated, t.ocr_text, cv::Point(20, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 255, 0), thick);   // 绿
        }
    }

    return annotated;
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
