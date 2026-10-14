#include "feigangcao_pipeline.h"

#include <chrono>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

FeigangcaoPipeline::FeigangcaoPipeline(const FeigangcaoServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    detector_ = std::make_unique<Detector>(config_.det_model, "", dev_id);
    std::cout << "[OK] Feigangcao det model loaded: " << config_.det_model << std::endl;

    classifier_ = std::make_unique<Classifier>(config_.cls_model, config_.cls_label, dev_id);
    std::cout << "[OK] Feigangcao cls model loaded: " << config_.cls_model << std::endl;

    warmup();
}

FeigangcaoPipelineResult FeigangcaoPipeline::process(const cv::Mat& image,
                                                     const std::string& station_id,
                                                     bool verbose)
{
    FeigangcaoPipelineResult result;

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- 定位模型 ----
    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    detector_->process(det_imgs, det_results);
    DetectionResult det_result = det_results.empty() ? DetectionResult{} : det_results[0];

    const std::vector<Detection>& dets = det_result.detections;

    if (verbose) {
        std::cout << "[DEBUG] station=" << station_id
                  << " det detections: " << dets.size() << std::endl;
        for (int i = 0; i < (int)dets.size(); i++) {
            const auto& d = dets[i];
            std::cout << "  det[" << i << "] class=" << d.class_name
                      << " conf=" << d.confidence
                      << " bbox=(" << d.bbox.x << "," << d.bbox.y
                      << "," << d.bbox.width << "," << d.bbox.height
                      << ")" << std::endl;
        }
    }

    // 取置信度最高的框
    int best = -1;
    for (int i = 0; i < (int)dets.size(); i++) {
        if (best < 0 || dets[i].confidence > dets[best].confidence) {
            best = i;
        }
    }

    if (best < 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }

    const Detection& sel = dets[best];
    cv::Rect roi = InferHelper::safeROI(
        sel.bbox.x, sel.bbox.y, sel.bbox.width, sel.bbox.height,
        image.cols, image.rows);
    if (roi.area() <= 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }

    result.detected = true;
    result.roi = roi;
    result.det_confidence = sel.confidence;

    if (verbose) {
        std::cout << "  best det[" << best << "] roi=(" << roi.x << "," << roi.y
                  << "," << roi.width << "," << roi.height << ")" << std::endl;
    }

    // ---- 分类模型：定位框裁剪送检 ----
    cv::Mat crop = image(roi).clone();
    if (crop.channels() == 1) {
        cv::Mat crop_bgr;
        cv::cvtColor(crop, crop_bgr, cv::COLOR_GRAY2BGR);
        crop = crop_bgr;
    }
    result.det_crop = crop.clone();

    std::vector<cv::Mat> cls_imgs = {crop};
    std::vector<ClassificationResult> cls_results;
    classifier_->process(cls_imgs, cls_results);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!cls_results.empty()) {
        result.result = cls_results[0].class_name;
        result.cls_confidence = cls_results[0].confidence;
    }

    if (verbose) {
        std::cout << "  cls result=" << result.result
                  << " conf=" << result.cls_confidence << std::endl;
    }

    result.annotated_image = createAnnotatedImage(image, result);
    return result;
}

cv::Mat FeigangcaoPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const FeigangcaoPipelineResult& result)
{
    cv::Mat annotated = src_img.clone();

    // 定位框：绿色，左上角写置信度
    if (result.detected) {
        cv::rectangle(annotated, result.roi, cv::Scalar(0, 255, 0), 4);
        std::string label = cv::format("%.2f", result.det_confidence);
        cv::putText(annotated, label,
                    cv::Point(result.roi.x, result.roi.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    }

    // 最终分类结果写在原图左上角，红色字
    if (!result.result.empty()) {
        cv::putText(annotated, result.result, cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 4.0, cv::Scalar(0, 0, 255), 8);
    }

    return annotated;
}

void FeigangcaoPipeline::warmup()
{
    std::cout << "[INFO] Warming up feigangcao models (10 iterations)..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    for (int i = 0; i < 10; i++) {
        std::vector<DetectionResult> det_results;
        detector_->process(dummy, det_results);
    }
    std::cout << "[OK] Detector warmed up (10 iterations)" << std::endl;

    for (int i = 0; i < 10; i++) {
        std::vector<ClassificationResult> cls_results;
        classifier_->process(dummy, cls_results);
    }
    std::cout << "[OK] Classifier warmed up (10 iterations)" << std::endl;
}

} // namespace Pipeline
} // namespace JHDeepCore
