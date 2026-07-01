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

            // ===== Step 6: 遍历分割实例，收集字符裁剪框 (boundingRect 轴对齐=水平) =====
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

                // 从 mask 提取最小外接矩形
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(char_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (contours.empty()) continue;

                cv::Rect char_bbox_local = cv::boundingRect(contours[0]);
                char_bbox_local = char_bbox_local & cv::Rect(0, 0, billet_img.cols, billet_img.rows);
                if (char_bbox_local.area() <= 0) continue;

                // 裁剪字符图像
                cv::Mat char_img = billet_img(char_bbox_local).clone();
                if (char_img.empty()) continue;

                cv::Mat char_img_bgr;
                if (char_img.channels() == 1) {
                    cv::cvtColor(char_img, char_img_bgr, cv::COLOR_GRAY2BGR);
                } else {
                    char_img_bgr = char_img;
                }

                CharCrop crop;
                crop.bbox_local = char_bbox_local;
                crop.mask = char_mask;
                crop.img_bgr = char_img_bgr;
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

    return annotated;
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
