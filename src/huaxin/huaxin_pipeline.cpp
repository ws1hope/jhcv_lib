#include "huaxin_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

HuaxinPipeline::HuaxinPipeline(const HuaxinServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det1_ = std::make_unique<Detector>(config_.det1_model, "", dev_id);
    std::cout << "[OK] Huaxin det1 model loaded: " << config_.det1_model << std::endl;

    det2_ = std::make_unique<Detector>(config_.det2_model, "", dev_id);
    std::cout << "[OK] Huaxin det2 model loaded: " << config_.det2_model << std::endl;

    warmup();
}

HuaxinPipelineResult HuaxinPipeline::process(const cv::Mat& image,
                                             const std::string& station_id,
                                             bool verbose)
{
    HuaxinPipelineResult result;

    // 按 station_id 选定当前工位的有效检测范围多边形（无匹配则空，表示不过滤）
    active_roi_poly_.clear();
    for (const auto& vr : config_.valid_rois) {
        if (vr.station_id == station_id) {
            for (const auto& p : vr.points) {
                active_roi_poly_.push_back(cv::Point(p[0], p[1]));
            }
            break;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- 第一个检测模型 ----
    std::vector<cv::Mat> det1_imgs = {image};
    std::vector<DetectionResult> det1_results;
    det1_->process(det1_imgs, det1_results);
    DetectionResult det1_result = det1_results.empty() ? DetectionResult{} : det1_results[0];

    std::vector<Detection> dets1 = det1_result.detections;
    result.det1_detections = dets1;

    if (verbose) {
        std::cout << "[DEBUG] det1 detections: " << dets1.size() << std::endl;
        for (int i = 0; i < (int)dets1.size(); i++) {
            const auto& d = dets1[i];
            std::cout << "  det1[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }
    }

    // 找最左侧的框（bbox.x 最小）
    int leftmost = -1;
    for (int i = 0; i < (int)dets1.size(); i++) {
        if (leftmost < 0 || dets1[i].bbox.x < dets1[leftmost].bbox.x) {
            leftmost = i;
        }
    }

    if (leftmost < 0) {
        // 第一个模型无输出，直接出结果图
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }

    result.leftmost_index = leftmost;
    const Detection& sel = dets1[leftmost];

    cv::Rect roi = InferHelper::safeROI(
        sel.bbox.x, sel.bbox.y, sel.bbox.width, sel.bbox.height,
        image.cols, image.rows);
    if (roi.area() <= 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }
    result.leftmost_roi = roi;

    if (verbose) {
        std::cout << "  leftmost det1[" << leftmost << "] roi=(" << roi.x << "," << roi.y
                  << "," << roi.width << "," << roi.height << ")" << std::endl;
    }

    // ---- 有效检测范围（当前工位 valid_roi 多边形）检查 ----
    // 最左框的中心点在多边形内才送入 det2
    if ((int)active_roi_poly_.size() >= 3) {
        cv::Point center(roi.x + roi.width / 2, roi.y + roi.height / 2);
        bool center_in_roi = cv::pointPolygonTest(active_roi_poly_, center, false) >= 0;

        if (verbose) {
            std::cout << "[DEBUG] valid_roi center=(" << center.x << "," << center.y
                      << ") in_roi=" << (center_in_roi ? "true" : "false") << std::endl;
        }

        if (!center_in_roi) {
            result.det2_skipped = true;
            auto t1 = std::chrono::high_resolution_clock::now();
            result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            result.annotated_image = createAnnotatedImage(image, result);
            return result;
        }
    }

    // ---- 第二个检测模型：最左框裁剪送检 ----
    cv::Mat crop = image(roi).clone();
    if (crop.channels() == 1) {
        cv::Mat crop_bgr;
        cv::cvtColor(crop, crop_bgr, cv::COLOR_GRAY2BGR);
        crop = crop_bgr;
    }

    std::vector<cv::Mat> det2_imgs = {crop};
    std::vector<DetectionResult> det2_results;
    det2_->process(det2_imgs, det2_results);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    DetectionResult det2_result = det2_results.empty() ? DetectionResult{} : det2_results[0];

    // 按从左到右排序，拼接结果符合阅读顺序
    std::vector<Detection> dets2 = det2_result.detections;
    std::sort(dets2.begin(), dets2.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.x < b.bbox.x;
        });

    if (verbose) {
        std::cout << "[DEBUG] det2 detections: " << dets2.size() << std::endl;
    }

    std::string all_results;
    for (int i = 0; i < (int)dets2.size(); i++) {
        const auto& d = dets2[i];

        if (verbose) {
            std::cout << "  det2[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }

        HuaxinDet2Target tgt;
        tgt.bbox_on_src = cv::Rect(d.bbox.x + roi.x, d.bbox.y + roi.y,
                                   d.bbox.width, d.bbox.height);
        tgt.class_name = d.class_name;
        tgt.confidence = d.confidence;
        result.det2_targets.push_back(tgt);

        all_results += d.class_name;
    }

    result.all_results = all_results;
    result.annotated_image = createAnnotatedImage(image, result);

    return result;
}

cv::Mat HuaxinPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const HuaxinPipelineResult& result)
{
    cv::Mat annotated = src_img.clone();

    // 有效检测范围多边形：蓝色线绘制
    if ((int)active_roi_poly_.size() >= 3) {
        cv::polylines(annotated, active_roi_poly_, true, cv::Scalar(255, 0, 0), 4);
    }

    // 因有效检测范围覆盖不足而跳过 det2：图上标注
    if (result.det2_skipped) {
        cv::putText(annotated, "OUT OF ROI", cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 0, 0), 5);
    }

    // 第一个检测模型：送入第二个模型的框为绿色，其余为红色，左上角写置信度
    for (int i = 0; i < (int)result.det1_detections.size(); i++) {
        const auto& d = result.det1_detections[i];
        bool is_leftmost = (i == result.leftmost_index);
        cv::Scalar color = is_leftmost ? cv::Scalar(0, 255, 0)   // 绿色
                                       : cv::Scalar(0, 0, 255);  // 红色
        cv::rectangle(annotated, d.bbox, color, 4);

        // 框的中心点，与框同色
        cv::Point center(d.bbox.x + d.bbox.width / 2, d.bbox.y + d.bbox.height / 2);
        cv::circle(annotated, center, 8, color, -1);

        std::string label = cv::format("%.2f", d.confidence);
        cv::putText(annotated, label,
                    cv::Point(d.bbox.x, d.bbox.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);
    }

    // 第二个检测模型：蓝色框，左上角写置信度
    for (const auto& t : result.det2_targets) {
        cv::rectangle(annotated, t.bbox_on_src, cv::Scalar(255, 0, 0), 4);

        std::string label = cv::format("%.2f", t.confidence);
        cv::putText(annotated, label,
                    cv::Point(t.bbox_on_src.x, t.bbox_on_src.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0), 3);
    }

    // 最终识别结果写在原图左上角，红色字
    if (!result.all_results.empty()) {
        cv::putText(annotated, result.all_results, cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 4.0, cv::Scalar(0, 0, 255), 8);
    }

    return annotated;
}

void HuaxinPipeline::warmup()
{
    std::cout << "[INFO] Warming up huaxin models (10 iterations)..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    for (int i = 0; i < 10; i++) {
        std::vector<DetectionResult> det1_results;
        det1_->process(dummy, det1_results);
    }
    std::cout << "[OK] Det1 warmed up (10 iterations)" << std::endl;

    for (int i = 0; i < 10; i++) {
        std::vector<DetectionResult> det2_results;
        det2_->process(dummy, det2_results);
    }
    std::cout << "[OK] Det2 warmed up (10 iterations)" << std::endl;
}

} // namespace Pipeline
} // namespace JHDeepCore
