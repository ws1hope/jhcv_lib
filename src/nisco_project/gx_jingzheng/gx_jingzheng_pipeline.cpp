#include "gx_jingzheng_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

namespace JHDeepCore {
namespace Pipeline {

namespace {

// 从 seg yaml 抽取 class_names；失败时回退到 [back_ground, gangbiao, zifu]
std::vector<std::string> loadSegClassNames(const std::string& yaml_path)
{
    std::vector<std::string> names;
    if (yaml_path.empty()) return names;
    try {
        YAML::Node node = YAML::LoadFile(yaml_path);
        if (node["class_names"]) {
            names = node["class_names"].as<std::vector<std::string>>();
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Failed to parse seg label yaml \"" << yaml_path
                  << "\": " << e.what() << std::endl;
    }
    return names;
}

int classIdByName(const std::vector<std::string>& names, const std::string& target)
{
    for (int i = 0; i < (int)names.size(); i++) {
        if (names[i] == target) return i;
    }
    return -1;
}

cv::Mat extractClassBinary(const cv::Mat& seg_mask, int class_id, const cv::Size& dst_size)
{
    if (seg_mask.empty() || class_id < 0) {
        return cv::Mat();
    }
    cv::Mat bin = (seg_mask == class_id);
    if (bin.size() != dst_size) {
        cv::resize(bin, bin, dst_size, 0, 0, cv::INTER_NEAREST);
    }
    return bin;
}

} // namespace

GxJingzhengPipeline::GxJingzhengPipeline(const GxJingzhengServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.dingwei_model, config_.dingwei_label, dev_id);
    std::cout << "[OK] Dingwei det model loaded: " << config_.dingwei_model << std::endl;

    seg_ = std::make_unique<Segmenter>(config_.seg_model, config_.seg_label, dev_id);
    std::cout << "[OK] Seg model loaded: " << config_.seg_model << std::endl;

    direction_cls_ = std::make_unique<Classifier>(config_.direction_cls_model, "", dev_id);
    std::cout << "[OK] Direction cls model loaded: " << config_.direction_cls_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] OCR model loaded: " << config_.ocr_model << std::endl;

    // 加载语义分割类别名 + 解析 zifu / gangbiao 的 class id
    seg_class_names_ = loadSegClassNames(config_.seg_label);
    if (seg_class_names_.empty()) {
        // 与 models/gx_xm/pm_yb_dw.yaml 对齐的默认顺序
        seg_class_names_ = {"back_ground", "gangbiao", "zifu"};
    }
    zifu_class_id_ = classIdByName(seg_class_names_, config_.zifu_class_name);
    gangbiao_class_id_ = classIdByName(seg_class_names_, config_.gangbiao_class_name);
    std::cout << "[OK] Seg class ids resolved: " << config_.zifu_class_name
              << "=" << zifu_class_id_ << ", " << config_.gangbiao_class_name
              << "=" << gangbiao_class_id_ << std::endl;

    // gangbiao 分支：可选构建 tiebiao 子 pipeline（如果配了 tiebiao_config）
    if (!config_.tiebiao_config.empty()) {
        TiebiaoServerConfig tcfg = FileHelper::loadTiebiaoConfig(config_.tiebiao_config);
        TiebiaoConfig pcfg;
        pcfg.label_seg_model = tcfg.label_seg_model;
        pcfg.char_seg_model = tcfg.char_seg_model;
        pcfg.ocr_model = tcfg.ocr_model;
        pcfg.ocr_label = tcfg.ocr_label;
        pcfg.direction_cls_model = tcfg.direction_cls_model;
        pcfg.device = config_.device.empty() ? tcfg.device : config_.device;
        tiebiao_pipeline_ = std::make_unique<TiebiaoPipeline>(pcfg);
        std::cout << "[OK] Tiebiao sub-pipeline loaded from: " << config_.tiebiao_config << std::endl;
    } else {
        std::cout << "[WARN] tiebiao_config not set; gangbiao branch will be unavailable." << std::endl;
    }

    warmup();
}

void GxJingzhengPipeline::warmup()
{
    std::cout << "[INFO] Warming up gx_jingzheng models..." << std::endl;
    std::vector<cv::Mat> dummy_imgs = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det_res;
    det_->process(dummy_imgs, det_res);
    std::cout << "[OK] Dingwei det warmed up" << std::endl;

    std::vector<SegmentationResult> seg_res;
    seg_->process(dummy_imgs, seg_res);
    std::cout << "[OK] Seg warmed up" << std::endl;

    std::vector<ClassificationResult> cls_res;
    direction_cls_->process(dummy_imgs, cls_res);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::vector<OCRResult> ocr_res;
    ocr_->process(dummy_imgs, ocr_res);
    std::cout << "[OK] OCR warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

std::string GxJingzhengPipeline::decideBranch(const cv::Mat& crop, cv::Mat& seg_mask_full_size)
{
    seg_mask_full_size = cv::Mat();
    if (crop.empty()) return "";

    std::vector<cv::Mat> imgs = {crop};
    std::vector<SegmentationResult> results;
    seg_->process(imgs, results);
    if (results.empty() || results[0].segmentation_mask.empty()) return "";

    cv::Mat mask = results[0].segmentation_mask;
    if (mask.size() != crop.size()) {
        cv::resize(mask, seg_mask_full_size, crop.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        seg_mask_full_size = mask;
    }

    // 统计 zifu / gangbiao 的前景像素数，多者胜
    long zifu_pixels = 0;
    long gangbiao_pixels = 0;
    if (zifu_class_id_ >= 0) {
        zifu_pixels = cv::countNonZero(seg_mask_full_size == zifu_class_id_);
    }
    if (gangbiao_class_id_ >= 0) {
        gangbiao_pixels = cv::countNonZero(seg_mask_full_size == gangbiao_class_id_);
    }

    if (zifu_pixels <= 0 && gangbiao_pixels <= 0) return "";
    return (zifu_pixels >= gangbiao_pixels) ? config_.zifu_class_name
                                            : config_.gangbiao_class_name;
}

int GxJingzhengPipeline::classifyDirection(const std::vector<cv::Mat>& char_images)
{
    if (char_images.empty()) return 0;
    int count_0 = 0, count_180 = 0;
    for (auto& img : char_images) {
        if (img.empty()) continue;
        cv::Mat bgr;
        if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        else bgr = img;

        std::vector<cv::Mat> imgs = {bgr};
        std::vector<ClassificationResult> results;
        direction_cls_->process(imgs, results);
        if (!results.empty()) {
            if (results[0].class_id == 0) count_0++;
            else count_180++;
        }
    }
    return (count_0 >= count_180) ? 0 : 180;
}

bool GxJingzhengPipeline::handleZifuBranch(const cv::Mat& crop,
                                            const cv::Mat& zifu_binary,
                                            GxJingzhengPipelineResult& result,
                                            bool verbose)
{
    if (zifu_binary.empty() || cv::countNonZero(zifu_binary) <= 0) {
        if (verbose) std::cout << "[DEBUG] zifu mask empty" << std::endl;
        return false;
    }

    // 连通域 → 每个连通域作为一个字符
    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(
        zifu_binary, labels, stats, centroids, 8, CV_32S);
    if (num_components <= 1) {
        if (verbose) std::cout << "[DEBUG] zifu connected components = 0" << std::endl;
        return false;
    }

    // 过滤掉过小的连通域（噪声）；阈值取整张 crop 面积的 0.01%
    const int min_area = std::max(20, crop.cols * crop.rows / 10000);

    std::vector<CharAngleInfo> char_infos;
    for (int comp = 1; comp < num_components; comp++) {
        int area = stats.at<int>(comp, cv::CC_STAT_AREA);
        if (area < min_area) continue;

        int x = stats.at<int>(comp, cv::CC_STAT_LEFT);
        int y = stats.at<int>(comp, cv::CC_STAT_TOP);
        int w = stats.at<int>(comp, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(comp, cv::CC_STAT_HEIGHT);

        cv::Rect safe = ImageHelper::safeClampROI(x, y, w, h, crop.cols, crop.rows);
        if (safe.area() <= 0) continue;

        // 该连通域在局部 ROI 内的二值 mask
        cv::Mat comp_mask = (labels(safe) == comp);
        if (cv::countNonZero(comp_mask) <= 0) continue;

        MinAreaRectResult mr = ImageHelper::computeMinAreaRect(comp_mask);

        CharAngleInfo info;
        info.angle = mr.angle;
        info.center_x = mr.center_x;
        info.center_y = mr.center_y;
        info.class_id = 0;
        info.bbox_offset = cv::Point2i(safe.x, safe.y);
        info.height = mr.height;
        for (int j = 0; j < 4; j++) info.corners[j] = mr.corners[j];
        char_infos.push_back(info);
    }

    if (char_infos.empty()) {
        if (verbose) std::cout << "[DEBUG] no zifu chars after filtering" << std::endl;
        return false;
    }

    // 取最高字符的角度作为整体旋转角，以所有字符的中心点平均作为旋转中心
    ImageHelper::sortByHeight(char_infos);
    float angle = char_infos.back().angle;

    int sum_x = 0, sum_y = 0;
    for (auto& info : char_infos) {
        sum_x += info.center_x + info.bbox_offset.x;
        sum_y += info.center_y + info.bbox_offset.y;
    }
    int center_x = sum_x / static_cast<int>(char_infos.size());
    int center_y = sum_y / static_cast<int>(char_infos.size());

    // 与 tiebiao_pipeline 一致：先 zero-pad，再围绕扩边后的中心点旋转
    const int bias = 30;
    cv::Mat padded = cv::Mat::zeros(
        crop.rows + 2 * bias, crop.cols + 2 * bias, CV_8UC3);
    crop.copyTo(padded(cv::Rect(bias, bias, crop.cols, crop.rows)));

    int x_off = 0, y_off = 0;
    cv::Mat rotated = ImageHelper::rotateImageAroundPoint(
        padded, center_x + bias, center_y + bias, angle, x_off, y_off);

    std::vector<CharCropInfo> char_crops = ImageHelper::rotateAndRemapBBoxes(
        char_infos, cv::Point2f(static_cast<float>(center_x),
                                 static_cast<float>(center_y)),
        angle, rotated, bias, bias, x_off, y_off);

    if (char_crops.empty()) {
        if (verbose) std::cout << "[DEBUG] rotateAndRemapBBoxes returned 0 chars" << std::endl;
        return false;
    }

    // 方向分类
    std::vector<cv::Mat> char_imgs_before;
    char_imgs_before.reserve(char_crops.size());
    for (auto& c : char_crops) char_imgs_before.push_back(c.image);
    int dir_flag = classifyDirection(char_imgs_before);

    if (verbose) std::cout << "[DEBUG] zifu direction flag = " << dir_flag << std::endl;

    // 按方向翻转后送 OCR，拼接最终文本
    std::string concatenated;
    result.chars.clear();
    result.chars.reserve(char_crops.size());

    for (auto& crop_info : char_crops) {
        GxJingzhengCharInfo cinfo;
        cinfo.bbox = crop_info.bbox;
        cinfo.image_before_flip = crop_info.image.clone();

        cv::Mat send_img;
        if (dir_flag == 180 && !crop_info.image.empty()) {
            cv::flip(crop_info.image, send_img, -1);
        } else {
            send_img = crop_info.image;
        }
        cinfo.image_after_flip = send_img.clone();

        if (!send_img.empty()) {
            cv::Mat bgr;
            if (send_img.channels() == 1)
                cv::cvtColor(send_img, bgr, cv::COLOR_GRAY2BGR);
            else
                bgr = send_img;

            std::vector<cv::Mat> ocr_imgs = {bgr};
            std::vector<OCRResult> ocr_results;
            ocr_->process(ocr_imgs, ocr_results);
            if (!ocr_results.empty() && !ocr_results[0].boxes.empty()) {
                cinfo.ocr_text = ocr_results[0].boxes[0].text;
            }
        }

        if (verbose) {
            std::cout << "[DEBUG]   char text=\"" << cinfo.ocr_text << "\"" << std::endl;
        }

        concatenated += cinfo.ocr_text;
        result.chars.push_back(std::move(cinfo));
    }

    result.rotated_crop = rotated;
    result.direction_flag = dir_flag;
    result.ocr_text = concatenated;
    return !concatenated.empty();
}

cv::Mat GxJingzhengPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const GxJingzhengPipelineResult& result)
{
    // gangbiao 分支：直接把 tiebiao 的标注图作为最终图
    if (result.branch == config_.gangbiao_class_name && !result.tiebiao_annotated.empty()) {
        return result.tiebiao_annotated.clone();
    }

    cv::Mat annotated = src_img.clone();

    // 第 1 层：所有 det 框（最左被选中的高亮）
    for (size_t i = 0; i < result.det_detections.size(); i++) {
        const auto& det = result.det_detections[i];
        bool is_chosen = (det.bbox == result.chosen_bbox);
        cv::Scalar color = is_chosen ? cv::Scalar(0, 255, 0) : cv::Scalar(120, 120, 120);
        int thickness = is_chosen ? 3 : 1;
        cv::rectangle(annotated, det.bbox, color, thickness);
        std::string label = "det" + std::to_string(i + 1) + " " + det.class_name
                            + cv::format(" %.2f", det.confidence);
        cv::putText(annotated, label,
                    cv::Point(det.bbox.x, std::max(0, det.bbox.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    // 第 2 层：分支标签与最终 OCR
    {
        std::string head = "branch=" + (result.branch.empty() ? std::string("?") : result.branch)
                           + " state=" + result.state_flag;
        cv::putText(annotated, head, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        if (!result.ocr_text.empty()) {
            cv::putText(annotated, "ocr: " + result.ocr_text,
                        cv::Point(20, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);
        }
    }

    // zifu 分支：底部拼一个矫正后图 + 字符片段长条
    if (result.branch == config_.zifu_class_name && !result.rotated_crop.empty()) {
        const int row_h = 60;
        const int margin = 15;
        const int gap = 15;
        const int thumb_height = 200;
        const int char_margin = 4;

        cv::Mat rotated = result.rotated_crop;
        if (result.direction_flag == 180) {
            cv::flip(rotated, rotated, -1);
        }
        float thumb_scale = static_cast<float>(thumb_height) / std::max(rotated.rows, 1);
        cv::Mat thumb;
        cv::resize(rotated, thumb, cv::Size(), thumb_scale, thumb_scale, cv::INTER_LINEAR);

        std::vector<cv::Mat> imgs_for_strip;
        for (auto& ch : result.chars) {
            if (!ch.image_after_flip.empty()) imgs_for_strip.push_back(ch.image_after_flip);
        }
        cv::Mat strip;
        if (!imgs_for_strip.empty()) {
            std::vector<cv::Mat> resized;
            int total_w = 0;
            for (auto& img : imgs_for_strip) {
                float s = static_cast<float>(row_h) / std::max(img.rows, 1);
                cv::Mat r;
                cv::resize(img, r, cv::Size(), s, s, cv::INTER_LINEAR);
                resized.push_back(r);
                total_w += r.cols + char_margin;
            }
            total_w = std::max(total_w - char_margin, 1);
            strip = cv::Mat::zeros(row_h, total_w, CV_8UC3);
            int cx = 0;
            for (auto& r : resized) {
                if (cx + r.cols > strip.cols) break;
                r.copyTo(strip(cv::Rect(cx, 0, r.cols, r.rows)));
                cx += r.cols + char_margin;
            }
        }

        int extra_h = std::max(thumb.rows, row_h) + 2 * margin;
        cv::Mat canvas = cv::Mat::zeros(annotated.rows + extra_h, annotated.cols, annotated.type());
        annotated.copyTo(canvas(cv::Rect(0, 0, annotated.cols, annotated.rows)));

        int y0 = annotated.rows + margin;
        if (!thumb.empty()) {
            int paste_w = std::min(thumb.cols, canvas.cols - margin);
            if (paste_w > 0) {
                thumb(cv::Rect(0, 0, paste_w, thumb.rows))
                    .copyTo(canvas(cv::Rect(margin, y0, paste_w, thumb.rows)));
            }
        }
        if (!strip.empty()) {
            int strip_x = margin + thumb.cols + gap;
            int avail = canvas.cols - strip_x - margin;
            if (avail > 0) {
                int paste_w = std::min(strip.cols, avail);
                int paste_y = y0 + (thumb_height - row_h) / 2;
                strip(cv::Rect(0, 0, paste_w, strip.rows))
                    .copyTo(canvas(cv::Rect(strip_x, paste_y, paste_w, strip.rows)));
            }
        }
        annotated = canvas;
    }

    return annotated;
}

GxJingzhengPipelineResult GxJingzhengPipeline::process(const cv::Mat& image,
                                                        int station_id,
                                                        const std::string& heat_number,
                                                        bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    GxJingzhengPipelineResult result;
    result.state_flag = "NG";

    // ===== 1) 定位检测 =====
    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_res = det_results.empty() ? DetectionResult{} : det_results[0];
    result.det_detections = det_res.detections;

    if (verbose) {
        std::cout << "[DEBUG] dingwei det = " << det_res.num_detections << std::endl;
    }
    if (det_res.num_detections <= 0) {
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }

    // ===== 2) 取最左检测框 =====
    int left_idx = 0;
    for (int i = 1; i < det_res.num_detections; i++) {
        if (det_res.detections[i].bbox.x < det_res.detections[left_idx].bbox.x) {
            left_idx = i;
        }
    }
    const auto& chosen = det_res.detections[left_idx];
    cv::Rect chosen_roi = InferHelper::safeROI(
        chosen.bbox.x, chosen.bbox.y, chosen.bbox.width, chosen.bbox.height,
        image.cols, image.rows);
    if (chosen_roi.area() <= 0) {
        if (verbose) std::cout << "[DEBUG] leftmost det bbox is invalid" << std::endl;
        result.annotated_image = createAnnotatedImage(image, result);
        return result;
    }
    result.chosen_bbox = chosen_roi;
    cv::Mat crop = image(chosen_roi).clone();
    if (verbose) {
        std::cout << "[DEBUG] leftmost det idx=" << left_idx
                  << " bbox=(" << chosen_roi.x << "," << chosen_roi.y
                  << "," << chosen_roi.width << "," << chosen_roi.height << ")" << std::endl;
    }

    // ===== 3) 语义分割决定分支 =====
    cv::Mat seg_mask;
    std::string branch = decideBranch(crop, seg_mask);
    if (verbose) std::cout << "[DEBUG] seg branch = \"" << branch << "\"" << std::endl;
    result.branch = branch;

    if (branch == config_.zifu_class_name) {
        // ===== 4a) zifu 分支：连通域 → 矫正 → 方向分类 → OCR =====
        cv::Mat zifu_binary = extractClassBinary(seg_mask, zifu_class_id_, crop.size());
        if (handleZifuBranch(crop, zifu_binary, result, verbose)) {
            result.state_flag = "OK";
        }
    } else if (branch == config_.gangbiao_class_name) {
        // ===== 4b) gangbiao 分支：转给 tiebiao =====
        if (!tiebiao_pipeline_) {
            std::cerr << "[ERROR] gangbiao branch but tiebiao_pipeline_ not initialized" << std::endl;
        } else {
            TiebiaoResult tres = tiebiao_pipeline_->process(crop, station_id, heat_number, verbose);
            result.ocr_text = tres.ocr_text;
            result.state_flag = tres.state_flag;
            result.tiebiao_annotated = tres.annotated_image;
        }
    }

    result.annotated_image = createAnnotatedImage(image, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (verbose) {
        std::cout << "[DEBUG] gx_jingzheng total inference time: " << ms << " ms" << std::endl;
    }
    return result;
}

} // namespace Pipeline
} // namespace JHDeepCore
