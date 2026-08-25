#include "fujian_pipeline.h"

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

FujianPipeline::FujianPipeline(const FujianStationConfig& station_config,
                               const std::string& device)
    : station_config_(station_config), device_(device)
{
    int dev_id = (device_ == "cuda" || device_ == "gpu") ? 0 : -1;

    seg_ = std::make_unique<InstanceSegmenter>(station_config_.det_model_path, "", dev_id);
    std::cout << "[OK] Fujian station " << station_config_.station_id
              << " seg model loaded: " << station_config_.det_model_path << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(station_config_.rec_model_path,
                                           station_config_.rec_label_path, dev_id);
    std::cout << "[OK] Fujian station " << station_config_.station_id
              << " OCR model loaded: " << station_config_.rec_model_path << std::endl;

    warmup();
}

FujianPipelineResult FujianPipeline::process(const cv::Mat& image, const cv::Rect& roi,
                                             int roi_index, const std::string& heat_number,
                                             bool verbose)
{
    FujianPipelineResult result;
    result.roi_index = roi_index;

    cv::Rect safe_roi = InferHelper::safeROI(roi.x, roi.y, roi.width, roi.height,
                                             image.cols, image.rows);
    if (safe_roi.area() <= 0) {
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }
    result.source_roi = roi;
    result.used_roi = safe_roi;

    if (verbose) {
        std::cout << "[DEBUG] roi=(" << safe_roi.x << "," << safe_roi.y
                  << "," << safe_roi.width << "," << safe_roi.height << ")" << std::endl;
    }

    // ---- ROI 裁剪 ----
    cv::Mat crop = image(safe_roi).clone();
    if (crop.channels() == 1) {
        cv::Mat crop_bgr;
        cv::cvtColor(crop, crop_bgr, cv::COLOR_GRAY2BGR);
        crop = crop_bgr;
    }
    result.roi_image = crop;

    // ---- 实例分割检出字符实例 ----
    std::vector<cv::Mat> seg_imgs = {crop};
    std::vector<InstanceSegmentationResult> seg_results;
    seg_->process(seg_imgs, seg_results);
    InstanceSegmentationResult seg_result = seg_results.empty() ? InstanceSegmentationResult{} : seg_results[0];

    // 按从左到右、从上到下（Z 型）排序后逐字符送 OCR
    std::vector<Detection> dets = sortZOrder(seg_result.detections);

    if (verbose) {
        std::cout << "[DEBUG] seg detections: " << dets.size() << std::endl;
    }

    // ---- 检出超过 2 框：只保留最右边的两个框作为后续判断依据 ----
    if (dets.size() > 2) {
        if (verbose) {
            std::cout << "[INFO] seg detections " << dets.size()
                      << " > 2, keep rightmost 2" << std::endl;
        }
        std::vector<Detection> by_x = dets;
        std::sort(by_x.begin(), by_x.end(),
            [](const Detection& a, const Detection& b) {
                return a.bbox.x > b.bbox.x;
            });
        dets = {by_x[0], by_x[1]};
        dets = sortZOrder(dets);
    }

    // ---- 分割恰好 2 框且宽度差 < 30px：定位异常，跳过 OCR，直接用用户输入炉号 ----
    if (dets.size() == 2 &&
        std::abs(dets[0].bbox.width - dets[1].bbox.width) < 30) {
        if (verbose) {
            std::cout << "[INFO] two seg detections with width diff < 30px ("
                      << dets[0].bbox.width << " vs " << dets[1].bbox.width
                      << "), use input heat_number: " << heat_number << std::endl;
        }
        for (const auto& d : dets) {
            FujianCharResult chr;
            chr.bbox_on_src = cv::Rect(d.bbox.x + safe_roi.x, d.bbox.y + safe_roi.y,
                                       d.bbox.width, d.bbox.height);
            chr.class_name = d.class_name;
            chr.confidence = d.confidence;
            result.chars.push_back(chr);
        }
        result.full_text = heat_number;
        result.annotated_image = createAnnotatedImage(image, result);
        // 炉号分两行显示：11 位第一行 6 位，其他第一行 5 位，其余放第二行
        int first_line = (heat_number.size() == 11) ? 6 : 5;
        std::string line1 = heat_number.substr(0, first_line);
        cv::putText(result.annotated_image, line1, cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
        if (heat_number.size() > (size_t)first_line) {
            std::string line2 = heat_number.substr(first_line);
            cv::putText(result.annotated_image, line2, cv::Point(10, 145),
                        cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
        }
        return result;
    }

    for (int i = 0; i < (int)dets.size(); i++) {
        const auto& d = dets[i];

        if (verbose) {
            std::cout << "  seg[" << i << "] class=" << d.class_name
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

        FujianCharResult chr;
        chr.bbox_on_src = cv::Rect(char_roi.x + safe_roi.x, char_roi.y + safe_roi.y,
                                   char_roi.width, char_roi.height);
        chr.class_name = d.class_name;
        chr.confidence = d.confidence;
        chr.ocr_text = text;
        chr.image = char_bgr;
        result.chars.push_back(chr);

        result.full_text += text;
    }

    result.annotated_image = createAnnotatedImage(image, result);

    return result;
}

cv::Mat FujianPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const FujianPipelineResult& result)
{
    cv::Mat annotated = src_img.clone();

    // 使用的 ROI：绿色框，左上角标配置第一个 list 坐标，右下角标第二个 list 坐标
    if (result.used_roi.area() > 0) {
        const cv::Rect& box = result.used_roi;
        const cv::Rect& src_roi = result.source_roi;
        cv::rectangle(annotated, box, cv::Scalar(0, 255, 0), 4);

        std::string tl_label = cv::format("(%d,%d)", src_roi.x, src_roi.y);
        cv::putText(annotated, tl_label,
                    cv::Point(box.x + 8, box.y - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

        std::string br_label = cv::format("(%d,%d)",
                                          src_roi.x + src_roi.width,
                                          src_roi.y + src_roi.height);
        int baseline = 0;
        cv::Size br_size = cv::getTextSize(br_label, cv::FONT_HERSHEY_SIMPLEX,
                                           1.2, 3, &baseline);
        cv::putText(annotated, br_label,
                    cv::Point(box.x + box.width - br_size.width - 8,
                              box.y + box.height + br_size.height + 12),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    }

    // 字符实例：蓝色框；左侧逐行写序号 + OCR 文字
    int text_y = 70;
    for (int i = 0; i < (int)result.chars.size(); i++) {
        const auto& c = result.chars[i];
        cv::rectangle(annotated, c.bbox_on_src, cv::Scalar(255, 0, 0), 4);

        if (!c.ocr_text.empty()) {
            std::string disp = cv::format("%d: %s", i + 1, c.ocr_text.c_str());
            cv::putText(annotated, disp, cv::Point(10, text_y),
                        cv::FONT_HERSHEY_SIMPLEX, 2.5, cv::Scalar(0, 0, 255), 6);
            text_y += 75;
        }
    }

    return annotated;
}

void FujianPipeline::warmup()
{
    std::cout << "[INFO] Warming up fujian station " << station_config_.station_id
              << " models..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<InstanceSegmentationResult> seg_results;
    seg_->process(dummy, seg_results);
    std::cout << "[OK] Seg model warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(dummy, ocr_results);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;
}

} // namespace Pipeline
} // namespace JHDeepCore
