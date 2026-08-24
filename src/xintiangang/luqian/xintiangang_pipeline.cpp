#include "xintiangang_pipeline.h"

#include <algorithm>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

XintiangangPipeline::XintiangangPipeline(const XintiangangServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.det_model, "", dev_id);
    std::cout << "[OK] Xintiangang detect model loaded: " << config_.det_model << std::endl;

    // 字符方向分类：class_id 0=正向不翻转，1=翻转 180° 后再送 OCR；未配置 direction_cls_model 则跳过
    if (!config_.direction_cls_model.empty()) {
        direction_cls_ = std::make_unique<Classifier>(config_.direction_cls_model, "", dev_id);
        std::cout << "[OK] Xintiangang direction cls model loaded: " << config_.direction_cls_model << std::endl;
    } else {
        std::cout << "[INFO] Xintiangang direction cls model not configured, skip direction classification" << std::endl;
    }

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] Xintiangang OCR model loaded: " << config_.ocr_model << std::endl;

    warmup();
}

XintiangangPipelineResult XintiangangPipeline::process(const cv::Mat& image, bool verbose)
{
    XintiangangPipelineResult result;

    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_result = det_results.empty() ? DetectionResult{} : det_results[0];

    if (verbose) {
        std::cout << "[DEBUG] detections: " << det_result.num_detections << std::endl;
    }

    std::vector<Detection> dets = det_result.detections;
    // 按从左到右排序，拼接结果符合阅读顺序
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.x < b.bbox.x;
        });

    result.detections = dets;

    std::string all_results;
    for (int i = 0; i < (int)dets.size(); i++) {
        const auto& d = dets[i];

        if (verbose) {
            std::cout << "  det[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }

        cv::Rect roi = InferHelper::safeROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            image.cols, image.rows);
        if (roi.area() <= 0) continue;

        cv::Mat crop = image(roi).clone();
        if (crop.empty()) continue;

        cv::Mat crop_bgr;
        if (crop.channels() == 1) {
            cv::cvtColor(crop, crop_bgr, cv::COLOR_GRAY2BGR);
        } else {
            crop_bgr = crop;
        }

        // 字符方向分类：class_id 1 -> 翻转 180° 再送 OCR；0 或未启用方向分类 -> 原向送 OCR
        cv::Mat ocr_in = crop_bgr;
        if (direction_cls_) {
            std::vector<cv::Mat> cls_imgs = {crop_bgr};
            std::vector<ClassificationResult> cls_results;
            direction_cls_->process(cls_imgs, cls_results);
            int dir_flag = (!cls_results.empty() && cls_results[0].class_id == 1) ? 1 : 0;
            if (dir_flag == 1) {
                cv::flip(crop_bgr, ocr_in, -1);  // 翻转 180°
            }
            if (verbose) {
                std::cout << "    direction_cls=" << dir_flag
                          << (dir_flag == 1 ? " (flipped 180)" : "") << std::endl;
            }
        }

        std::vector<cv::Mat> ocr_imgs = {ocr_in};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);
        OCRResult ocr_result = ocr_results.empty() ? OCRResult{} : ocr_results[0];

        std::string text;
        for (const auto& box : ocr_result.boxes) {
            if (!box.text.empty()) text += box.text;
        }

        if (verbose) {
            std::cout << "    ocr_text=\"" << text << "\"" << std::endl;
        }

        XintiangangTargetResult tgt;
        tgt.bbox_on_src = roi;
        tgt.class_name = d.class_name;
        tgt.confidence = d.confidence;
        tgt.ocr_text = text;
        result.targets.push_back(tgt);

        if (!text.empty()) {
            all_results += text;
        }
    }

    result.all_results = all_results;
    result.annotated_image = createAnnotatedImage(image, result.targets);

    return result;
}

cv::Mat XintiangangPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<XintiangangTargetResult>& targets)
{
    cv::Mat annotated = src_img.clone();
    int text_y = 70;
    for (int i = 0; i < (int)targets.size(); i++) {
        const auto& t = targets[i];
        cv::rectangle(annotated, t.bbox_on_src, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("det%d %s %.2f", i + 1,
                                       t.class_name.c_str(), t.confidence);
        cv::putText(annotated, label,
                    cv::Point(t.bbox_on_src.x, t.bbox_on_src.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (!t.ocr_text.empty()) {
            std::string disp = cv::format("det%d: %s", i + 1, t.ocr_text.c_str());
            cv::putText(annotated, disp, cv::Point(10, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
            text_y += 75;
        }
    }
    return annotated;
}

void XintiangangPipeline::warmup()
{
    std::cout << "[INFO] Warming up xintiangang models..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det_results;
    det_->process(dummy, det_results);
    std::cout << "[OK] Detector warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(dummy, ocr_results);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;

    if (direction_cls_) {
        std::vector<ClassificationResult> cls_results;
        direction_cls_->process(dummy, cls_results);
        std::cout << "[OK] Direction classifier warmed up" << std::endl;
    }
}

} // namespace Pipeline
} // namespace JHDeepCore
