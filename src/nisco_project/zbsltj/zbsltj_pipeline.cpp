#include "zbsltj_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <regex>
#include <set>

#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

ZbsltjPipeline::ZbsltjPipeline(const ZbsltjConfig& config)
    : config_(config)
{
    int dev_id = (config.device == "cuda" || config.device == "gpu") ? 0 : -1;

    billet_det_ = std::make_unique<Detector>(config.billet_det_model, "", dev_id);
    std::cout << "[OK] Billet det model loaded: " << config.billet_det_model << std::endl;

    char_seg_ = std::make_unique<InstanceSegmenter>(config.char_seg_model, "", dev_id);
    std::cout << "[OK] Char seg model loaded: " << config.char_seg_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config.ocr_model, config.ocr_label, dev_id);
    std::cout << "[OK] OCR model loaded: " << config.ocr_model << std::endl;

    warmup();
}

void ZbsltjPipeline::warmup()
{
    std::cout << "[INFO] Warming up models..." << std::endl;
    cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Mat> imgs = {dummy};

    std::vector<DetectionResult> det_results;
    billet_det_->process(imgs, det_results);
    std::cout << "[OK] Billet det warmed up" << std::endl;

    std::vector<InstanceSegmentationResult> seg_results;
    char_seg_->process(imgs, seg_results);
    std::cout << "[OK] Char seg warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(imgs, ocr_results);
    std::cout << "[OK] OCR warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

// step 1: billet detection
std::vector<std::pair<cv::Mat, cv::Rect>> ZbsltjPipeline::detectBillets(const cv::Mat& image)
{
    std::vector<cv::Mat> imgs = {image};
    std::vector<DetectionResult> results;
    billet_det_->process(imgs, results);

    std::vector<std::pair<cv::Mat, cv::Rect>> billet_rois;
    if (results.empty() || results[0].num_detections <= 0) return billet_rois;

    auto& det = results[0];
    std::vector<Detection> sorted_dets = det.detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.bbox.x < b.bbox.x;
              });

    for (auto& d : sorted_dets) {
        if (d.bbox.width <= 100 || d.bbox.height <= 100) continue;

        cv::Rect safe = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            image.cols, image.rows);
        if (safe.area() <= 0) continue;

        cv::Mat roi = image(safe).clone();
        billet_rois.emplace_back(roi, safe);
    }

    return billet_rois;
}

// step 2: char segmentation + rotation correction
bool ZbsltjPipeline::segmentAndRotateChars(const cv::Mat& billet_roi,
                                            cv::Mat& rotated_image,
                                            std::vector<CharCropInfo>& char_crops)
{
    char_crops.clear();

    std::vector<cv::Mat> imgs = {billet_roi};
    std::vector<InstanceSegmentationResult> results;
    char_seg_->process(imgs, results);

    if (results.empty() || results[0].num_detections <= 0) return false;

    auto& det = results[0];
    std::vector<CharAngleInfo> char_infos;

    for (int i = 0; i < det.num_detections; i++) {
        auto& d = det.detections[i];
        auto& mask_full = det.masks[i];

        cv::Rect safe = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            billet_roi.cols, billet_roi.rows);
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

    if (char_infos.empty()) return false;

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
        billet_roi.rows + 2 * bias, billet_roi.cols + 2 * bias, CV_8UC3);
    billet_roi.copyTo(padded(cv::Rect(bias, bias, billet_roi.cols, billet_roi.rows)));

    int x_off = 0, y_off = 0;
    rotated_image = ImageHelper::rotateImageAroundPoint(
        padded, center_x + bias, center_y + bias, angle, x_off, y_off);

    char_crops = ImageHelper::rotateAndRemapBBoxes(
        char_infos, cv::Point2f(static_cast<float>(center_x),
                                 static_cast<float>(center_y)),
        angle, rotated_image, bias, bias, x_off, y_off);

    return !char_crops.empty();
}

// step 3: OCR
void ZbsltjPipeline::recognizeFragments(const std::vector<CharCropInfo>& crops,
                                         std::vector<std::string>& texts)
{
    texts.clear();

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

        std::vector<cv::Mat> ocr_imgs = {bgr};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);

        std::string text;
        if (!ocr_results.empty() && !ocr_results[0].boxes.empty()) {
            text = ocr_results[0].boxes[0].text;
        }
        texts.push_back(text);
    }
}

// step 4: sort by y (top-down)
void ZbsltjPipeline::sortByY(std::vector<CharCropInfo>& crops,
                               std::vector<std::string>& texts) const
{
    if (texts.empty()) return;

    int n = static_cast<int>(crops.size());
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return crops[a].center_y < crops[b].center_y;
    });

    std::vector<CharCropInfo> new_crops;
    std::vector<std::string> new_texts;
    new_crops.reserve(n);
    new_texts.reserve(n);

    for (int k : idx) {
        new_crops.push_back(crops[k]);
        new_texts.push_back(texts[k]);
    }

    crops = std::move(new_crops);
    texts = std::move(new_texts);
}

// char-coverage similarity (= calculateSimilarity)
double ZbsltjPipeline::calcSimilarity(const std::vector<std::string>& fragments,
                                        const std::string& target) const
{
    if (target.empty()) return 0.0;

    std::vector<bool> covered(target.length(), false);

    for (auto& frag : fragments) {
        if (frag.empty()) continue;
        size_t searchPos = 0;
        while ((searchPos = target.find(frag, searchPos)) != std::string::npos) {
            for (size_t i = 0; i < frag.length(); ++i) {
                if (searchPos + i < target.length()) {
                    covered[searchPos + i] = true;
                }
            }
            searchPos += 1;
        }
    }

    int coveredCount = static_cast<int>(std::count(covered.begin(), covered.end(), true));
    return static_cast<double>(coveredCount) / target.length();
}

// penma format post-processing
std::string ZbsltjPipeline::formatPenmaResult(const std::string& str)
{
    std::smatch matches;

    static std::regex pat_2_3(R"((\w{6})#(\w{2})#(\w{3}))");
    if (std::regex_match(str, matches, pat_2_3)) {
        return matches[1].str() + matches[2].str() + " " + matches[3].str();
    }

    static std::regex pat_3_2(R"((\w{6})#(\w{3})#(\w{2}))");
    if (std::regex_match(str, matches, pat_3_2)) {
        return matches[1].str() + matches[3].str() + " " + matches[2].str();
    }

    static std::regex pat_rot_2_3(R"((\w{3})#(\w{2})#(\w{6}))");
    if (std::regex_match(str, matches, pat_rot_2_3)) {
        return matches[3].str() + matches[2].str() + " " + matches[1].str();
    }

    static std::regex pat_rot_3_2(R"((\w{2})#(\w{3})#(\w{6}))");
    if (std::regex_match(str, matches, pat_rot_3_2)) {
        return matches[3].str() + matches[1].str() + " " + matches[2].str();
    }

    static std::regex pat_6_5(R"((\w{6})#(\w{2})(\w{3}))");
    if (std::regex_match(str, matches, pat_6_5)) {
        return matches[1].str() + matches[2].str() + " " + matches[3].str();
    }

    return str;
}

// compare first 8 chars
bool ZbsltjPipeline::matchFirstEight(const std::string& str1, const std::string& str2)
{
    if (str1.size() < 8 || str2.size() < 8) return false;
    return str1.compare(0, 8, str2, 0, 8) == 0;
}

// main process entry
std::vector<ZbsltjBilletResult> ZbsltjPipeline::process(
    const cv::Mat& image,
    const std::vector<std::string>& candidate_heats,
    const std::vector<std::string>& candidate_pdis,
    std::string& current_heat,
    int& seq_counter,
    bool verbose)
{
    std::vector<ZbsltjBilletResult> all_results;

    if (image.empty()) return all_results;

    auto t_start = std::chrono::high_resolution_clock::now();
    last_timing_ = ZbsltjTiming{};

    // step 1: billet detection
    auto t_det0 = std::chrono::high_resolution_clock::now();
    auto billets = detectBillets(image);
    auto t_det1 = std::chrono::high_resolution_clock::now();
    double det_ms = std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();

    if (billets.empty()) {
        if (verbose) std::cout << "[DEBUG] No billets detected" << std::endl;
        last_timing_.det_ms = det_ms;
        last_timing_.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        return all_results;
    }
    if (verbose) std::cout << "[DEBUG] Detected " << billets.size() << " billets" << std::endl;

    // prepare candidate heats (max 3)
    std::string heat0, heat1, heat2;
    if (candidate_heats.size() > 0) heat0 = candidate_heats[0];
    if (candidate_heats.size() > 1) heat1 = candidate_heats[1];
    if (candidate_heats.size() > 2) heat2 = candidate_heats[2];

    std::string pdi0, pdi1, pdi2;
    if (candidate_pdis.size() > 0) pdi0 = candidate_pdis[0];
    if (candidate_pdis.size() > 1) pdi1 = candidate_pdis[1];
    if (candidate_pdis.size() > 2) pdi2 = candidate_pdis[2];

    double seg_ms = 0.0, ocr_ms = 0.0;
    for (int kk = 0; kk < (int)billets.size(); kk++) {
        auto& [billet_roi, bbox] = billets[kk];

        if (verbose) std::cout << "[DEBUG] Processing billet " << kk << std::endl;

        ZbsltjBilletResult result;
        result.bbox = bbox;

        // step 2: char segmentation + rotation
        cv::Mat rotated_image;
        std::vector<CharCropInfo> char_crops;
        auto t_seg0 = std::chrono::high_resolution_clock::now();
        bool seg_ok = segmentAndRotateChars(billet_roi, rotated_image, char_crops);
        auto t_seg1 = std::chrono::high_resolution_clock::now();
        seg_ms += std::chrono::duration<double, std::milli>(t_seg1 - t_seg0).count();

        if (!seg_ok) {
            if (verbose) std::cout << "[DEBUG] No chars detected in billet " << kk << std::endl;
            result.success = false;
            all_results.push_back(result);
            continue;
        }
        result.rotated_image = rotated_image;
        result.char_crops = char_crops;

        if (verbose) std::cout << "[DEBUG] " << char_crops.size() << " chars detected" << std::endl;

        // step 3: OCR
        std::vector<std::string> texts;
        auto t_ocr0 = std::chrono::high_resolution_clock::now();
        recognizeFragments(char_crops, texts);
        auto t_ocr1 = std::chrono::high_resolution_clock::now();
        ocr_ms += std::chrono::duration<double, std::milli>(t_ocr1 - t_ocr0).count();

        // step 4: sort by y
        sortByY(char_crops, texts);
        result.rec_texts = texts;
        result.char_crops = char_crops;

        if (verbose) {
            for (int i = 0; i < (int)texts.size(); i++) {
                std::cout << "[DEBUG]   char[" << i << "] text=\"" << texts[i] << "\"" << std::endl;
            }
        }

        // step 5: heat number matching (calculateSimilarity)
        double sim0 = calcSimilarity(texts, heat0);
        double sim1 = calcSimilarity(texts, heat1);
        double sim2 = calcSimilarity(texts, heat2);

        std::vector<double> sims = {sim0, sim1, sim2};
        int max_idx = 0;
        for (int i = 1; i < 3; i++) {
            if (sims[i] > sims[max_idx]) max_idx = i;
        }

        if (verbose) {
            std::cout << "[DEBUG] sim0=" << sim0 << " sim1=" << sim1 << " sim2=" << sim2
                      << " max_idx=" << max_idx << std::endl;
        }

        bool keep_luhao = false;
        if (sim0 < 0.5 && sim1 < 0.5 && sim2 < 0.5) {
            keep_luhao = true;
            if (verbose) std::cout << "[DEBUG] All sims < 0.5, keep current heat" << std::endl;
        }

        std::string detected_heat;
        std::string detected_pdi;
        if (max_idx == 0) { detected_heat = heat0; detected_pdi = pdi0; }
        else if (max_idx == 1) { detected_heat = heat1; detected_pdi = pdi1; }
        else { detected_heat = heat2; detected_pdi = pdi2; }

        result.matched_heat = detected_heat;
        result.pdi_count = detected_pdi;

        // step 6: sequence logic
        if (current_heat.empty()) {
            current_heat = detected_heat;
        }

        if (!detected_heat.empty() && detected_heat != current_heat && !keep_luhao) {
            if (verbose) {
                std::cout << "[DEBUG] Heat changed: " << current_heat
                          << " -> " << detected_heat << ", reset seq to 1" << std::endl;
            }
            seq_counter = 1;
            current_heat = detected_heat;
        }

        result.seq_number = seq_counter;
        result.success = true;
        all_results.push_back(result);

        seq_counter++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    last_timing_.det_ms = det_ms;
    last_timing_.seg_ms = seg_ms;
    last_timing_.ocr_ms = ocr_ms;
    last_timing_.total_ms = total_ms;
    if (verbose) {
        std::cout << "[DEBUG] timing: det=" << det_ms << "ms seg=" << seg_ms
                  << "ms ocr=" << ocr_ms << "ms total=" << total_ms << "ms" << std::endl;
    }

    return all_results;
}

const ZbsltjTiming& ZbsltjPipeline::lastTiming() const
{
    return last_timing_;
}

// draw results on image
void ZbsltjPipeline::drawResults(cv::Mat& image,
                                   const std::vector<ZbsltjBilletResult>& results) const
{
    for (auto& item : results) {
        int cx = item.bbox.x;
        int cy = item.bbox.y;
        int w = item.bbox.width;
        int h = item.bbox.height;

        int tl_x = cx;
        int tl_y = cy;

        cv::Rect draw_rect(tl_x, tl_y, w, h);
        cv::rectangle(image, draw_rect, cv::Scalar(0, 255, 0), 2);

        std::string seq_str = std::to_string(item.seq_number);

        std::string text_to_show;
        for (int i = 0; i < (int)item.rec_texts.size(); i++) {
            if (i == 0) text_to_show += item.rec_texts[i];
            else text_to_show += "#" + item.rec_texts[i];
        }

        std::string reformatted = formatPenmaResult(text_to_show);
        bool draw_seq = matchFirstEight(item.matched_heat, reformatted);

        if (!text_to_show.empty()) {
            cv::Point text_pos(tl_x, tl_y - 5);
            cv::Point seq_pos(tl_x + 70, tl_y - 120);

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text_to_show, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
            cv::rectangle(image,
                cv::Rect(text_pos.x, text_pos.y - textSize.height,
                         textSize.width, textSize.height + baseline),
                cv::Scalar(0, 255, 0), cv::FILLED);

            cv::putText(image, reformatted, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            std::string show_luhao = "luhao:" + item.matched_heat;
            std::string show_pdi = "pdi:" + item.pdi_count;

            cv::Point luhao_pos(tl_x, tl_y - 80);
            cv::Point pdi_pos(tl_x, tl_y - 40);

            if (draw_seq) {
                cv::putText(image, seq_str, seq_pos,
                            cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 4);

                int baseline_l = 0;
                cv::Size ts_l = cv::getTextSize(show_luhao, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline_l);
                cv::rectangle(image,
                    cv::Rect(luhao_pos.x, luhao_pos.y - ts_l.height,
                             ts_l.width, ts_l.height + baseline_l),
                    cv::Scalar(255, 0, 0), cv::FILLED);
                cv::putText(image, show_luhao, luhao_pos,
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

                int baseline_p = 0;
                cv::Size ts_p = cv::getTextSize(show_pdi, cv::FONT_HERSHEY_SIMPLEX, 1, 1, &baseline_p);
                cv::rectangle(image,
                    cv::Rect(pdi_pos.x, pdi_pos.y - ts_p.height,
                             ts_p.width, ts_p.height + baseline_p),
                    cv::Scalar(255, 0, 0), cv::FILLED);
                cv::putText(image, show_pdi, pdi_pos,
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            }
        }
    }
}

} // namespace Pipeline
} // namespace JHDeepCore
