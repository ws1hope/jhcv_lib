#include "guokuache_pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

// Z 型排序：从左到右、从上到下。先按 y 分行（行判定阈值取平均框高的一半，
// 中心 y 之差超过阈值即视为新行），行内按 x 从小到大，最后按行序拼接
static std::vector<Detection> sortZOrder(const std::vector<Detection>& dets)
{
    if (dets.size() <= 1) return dets;

    float avg_h = 0.0f;
    for (const auto& d : dets) avg_h += (float)d.bbox.height;
    avg_h /= (float)dets.size();
    float row_tol = avg_h * 0.5f;

    std::vector<Detection> by_y = dets;
    std::sort(by_y.begin(), by_y.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.y < b.bbox.y;
        });

    std::vector<Detection> sorted;
    std::vector<Detection> row;
    float row_center_y = 0.0f;
    for (const auto& d : by_y) {
        float center_y = d.bbox.y + d.bbox.height / 2.0f;
        if (!row.empty() && std::fabs(center_y - row_center_y) > row_tol) {
            std::sort(row.begin(), row.end(),
                [](const Detection& a, const Detection& b) {
                    return a.bbox.x < b.bbox.x;
                });
            sorted.insert(sorted.end(), row.begin(), row.end());
            row.clear();
        }
        if (row.empty()) row_center_y = center_y;
        row.push_back(d);
    }
    if (!row.empty()) {
        std::sort(row.begin(), row.end(),
            [](const Detection& a, const Detection& b) {
                return a.bbox.x < b.bbox.x;
            });
        sorted.insert(sorted.end(), row.begin(), row.end());
    }
    return sorted;
}

GuokuachePipeline::GuokuachePipeline(const GuokuacheServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det1_ = std::make_unique<Detector>(config_.det1_model, "", dev_id);
    std::cout << "[OK] Guokuache det1 model loaded: " << config_.det1_model << std::endl;

    det2_ = std::make_unique<Detector>(config_.det2_model, "", dev_id);
    std::cout << "[OK] Guokuache det2 model loaded: " << config_.det2_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] Guokuache OCR model loaded: " << config_.ocr_model << std::endl;

    warmup();
}

GuokuachePipelineResult GuokuachePipeline::process(const cv::Mat& image, bool verbose)
{
    GuokuachePipelineResult result;

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
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }

    result.selected_index = leftmost;
    const Detection& sel = dets1[leftmost];

    cv::Rect roi = InferHelper::safeROI(
        sel.bbox.x, sel.bbox.y, sel.bbox.width, sel.bbox.height,
        image.cols, image.rows);
    if (roi.area() <= 0) {
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }
    result.selected_roi = roi;

    if (verbose) {
        std::cout << "  leftmost det1[" << leftmost << "] roi=(" << roi.x << "," << roi.y
                  << "," << roi.width << "," << roi.height << ")" << std::endl;
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
    DetectionResult det2_result = det2_results.empty() ? DetectionResult{} : det2_results[0];

    // 按从左到右、从上到下（Z 型）排序后逐框送 OCR
    std::vector<Detection> dets2 = sortZOrder(det2_result.detections);

    if (verbose) {
        std::cout << "[DEBUG] det2 detections: " << dets2.size() << std::endl;
    }

    for (int i = 0; i < (int)dets2.size(); i++) {
        const auto& d = dets2[i];

        if (verbose) {
            std::cout << "  det2[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }

        cv::Rect char_roi = InferHelper::safeROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            crop.cols, crop.rows);
        if (char_roi.area() <= 0) continue;

        cv::Mat char_crop = crop(char_roi).clone();
        if (char_crop.empty()) continue;

        cv::Mat char_bgr;
        if (char_crop.channels() == 1) {
            cv::cvtColor(char_crop, char_bgr, cv::COLOR_GRAY2BGR);
        } else {
            char_bgr = char_crop;
        }

        std::vector<cv::Mat> ocr_imgs = {char_bgr};
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

        GuokuacheTargetResult tgt;
        tgt.bbox_on_src = cv::Rect(char_roi.x + roi.x, char_roi.y + roi.y,
                                   char_roi.width, char_roi.height);
        tgt.class_name = d.class_name;
        tgt.confidence = d.confidence;
        tgt.ocr_text = text;
        result.targets.push_back(tgt);
    }

    result.annotated_image = createAnnotatedImage(image, result);

    return result;
}

cv::Mat GuokuachePipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const GuokuachePipelineResult& result)
{
    cv::Mat annotated = src_img.clone();

    // 第一个检测模型：送入第二个模型的框为绿色，其余为红色，左上角写置信度
    for (int i = 0; i < (int)result.det1_detections.size(); i++) {
        const auto& d = result.det1_detections[i];
        bool is_selected = (i == result.selected_index);
        cv::Scalar color = is_selected ? cv::Scalar(0, 255, 0)   // 绿色
                                       : cv::Scalar(0, 0, 255);  // 红色
        cv::rectangle(annotated, d.bbox, color, 4);

        std::string label = cv::format("%.2f", d.confidence);
        cv::putText(annotated, label,
                    cv::Point(d.bbox.x, d.bbox.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);
    }

    // 第二个检测模型：蓝色框，左上角写序号 + OCR 文字
    int text_y = 70;
    for (int i = 0; i < (int)result.targets.size(); i++) {
        const auto& t = result.targets[i];
        cv::rectangle(annotated, t.bbox_on_src, cv::Scalar(255, 0, 0), 4);

        std::string label = cv::format("%d %s %.2f", i + 1,
                                       t.class_name.c_str(), t.confidence);
        cv::putText(annotated, label,
                    cv::Point(t.bbox_on_src.x, t.bbox_on_src.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0), 3);

        if (!t.ocr_text.empty()) {
            std::string disp = cv::format("%d: %s", i + 1, t.ocr_text.c_str());
            cv::putText(annotated, disp, cv::Point(10, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
            text_y += 75;
        }
    }

    return annotated;
}

void GuokuachePipeline::warmup()
{
    std::cout << "[INFO] Warming up guokuache models..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det1_results;
    det1_->process(dummy, det1_results);
    std::cout << "[OK] Det1 warmed up" << std::endl;

    std::vector<DetectionResult> det2_results;
    det2_->process(dummy, det2_results);
    std::cout << "[OK] Det2 warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(dummy, ocr_results);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;
}

} // namespace Pipeline
} // namespace JHDeepCore
