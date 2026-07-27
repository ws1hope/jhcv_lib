#include "gx_jingzheng_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include <opencv2/dnn.hpp>
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

// 把一次 lastBatchTiming() 的结果累加进 per-model 累加器
void accumulateTiming(InferenceTiming& acc, const InferenceTiming& t)
{
    acc.count += t.count;
    acc.preprocess_ms += t.preprocess_ms;
    acc.tensor_ms += t.tensor_ms;
    acc.run_ms += t.run_ms;
    acc.h2d_ms += t.h2d_ms;
    acc.h2d_split = acc.h2d_split || t.h2d_split;
    if (acc.device.empty() && !t.device.empty()) acc.device = t.device;
}

} // namespace

GxJingzhengPipeline::GxJingzhengPipeline(const GxJingzhengServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.dingwei_model, config_.dingwei_label, dev_id);
    std::cout << "[OK] Dingwei det model loaded: " << config_.dingwei_model << std::endl;

    
    // 注意：InstanceSegmenter 的第二个参数被实现当作 "class_names 列表" 使用，
    // 不是 yaml 路径。这里传空，让 OnnxInference 自动从 <model>.yaml 读 class_names。
    // iou=0.9：seg 在分支判定前就跑完 NMS，无法按分支给不同阈值。故整体取宽松 0.9
    // 保留全部候选框（gangbiao 分支直接用），zifu 分支在 decideBranch() 末尾再按 0.45
    // 二次 NMS 去重叠框。
    seg_ = std::make_unique<InstanceSegmenter>(config_.seg_model, "", dev_id, "",
                                                 0.25f, 0.9f);
    std::cout << "[OK] Instance seg model loaded: " << config_.seg_model << std::endl;

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
        // gangbiao: 方向分类判为翻转时，置信度低于该值不翻转(独立 tiebiao/dispatch 不受影响)
        pcfg.dir_flip_conf_threshold = 0.95f;
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

    std::vector<InstanceSegmentationResult> seg_res;
    seg_->process(dummy_imgs, seg_res);
    std::cout << "[OK] Instance seg warmed up" << std::endl;

    std::vector<ClassificationResult> cls_res;
    direction_cls_->process(dummy_imgs, cls_res);
    std::cout << "[OK] Direction cls warmed up" << std::endl;

    std::vector<OCRResult> ocr_res;
    ocr_->process(dummy_imgs, ocr_res);
    std::cout << "[OK] OCR warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

std::string GxJingzhengPipeline::decideBranch(const cv::Mat& crop,
                                                std::vector<GxJingzhengSegInstance>& instances_out)
{
    instances_out.clear();
    if (crop.empty()) return "";

    std::vector<cv::Mat> imgs = {crop};
    std::vector<InstanceSegmentationResult> results;
    seg_->process(imgs, results);
    accumulateTiming(t_seg_, seg_->lastBatchTiming());
    if (results.empty() || results[0].num_detections <= 0) return "";

    const auto& ir = results[0];
    long zifu_pixels = 0;
    long gangbiao_pixels = 0;

    for (int i = 0; i < ir.num_detections; i++) {
        const auto& det = ir.detections[i];
        if ((int)ir.masks.size() <= i || ir.masks[i].empty()) continue;

        // 把 mask 统一到 crop 大小、CV_8UC1 二值
        cv::Mat mask_full;
        if (ir.masks[i].size() != crop.size()) {
            cv::resize(ir.masks[i], mask_full, crop.size(), 0, 0, cv::INTER_NEAREST);
        } else {
            mask_full = ir.masks[i];
        }
        cv::Mat mask_bin;
        if (mask_full.type() != CV_8UC1) {
            mask_full.convertTo(mask_bin, CV_8UC1);
        } else {
            mask_bin = mask_full.clone();
        }
        cv::threshold(mask_bin, mask_bin, 0, 255, cv::THRESH_BINARY);

        // 解析 class_name：优先用 det.class_name，若它不在已知类名表里则按 class_id 查表兜底
        auto isKnown = [&](const std::string& n) {
            for (auto& s : seg_class_names_) {
                if (s == n) return true;
            }
            return false;
        };
        std::string cls = det.class_name;
        if (!isKnown(cls) && det.class_id >= 0 &&
            det.class_id < (int)seg_class_names_.size()) {
            cls = seg_class_names_[det.class_id];
        }
        if (cls == "back_ground") continue;

        long cnt = cv::countNonZero(mask_bin);
        if (cnt <= 0) continue;
        if (cls == config_.zifu_class_name) zifu_pixels += cnt;
        else if (cls == config_.gangbiao_class_name) gangbiao_pixels += cnt;

        GxJingzhengSegInstance inst;
        inst.class_name = cls;
        inst.bbox = det.bbox;
        inst.mask = mask_bin;
        inst.confidence = det.confidence;
        instances_out.push_back(std::move(inst));
    }

    if (zifu_pixels <= 0 && gangbiao_pixels <= 0) return "";
    std::string branch = (zifu_pixels >= gangbiao_pixels) ? config_.zifu_class_name
                                                           : config_.gangbiao_class_name;

    // zifu 分支：对 zifu 类实例按 0.45 二次 NMS 去重叠框
    // （seg 整体用 0.9 兼容 gangbiao，故在此补严；gangbiao 分支不进此块，保持 0.9）
    if (branch == config_.zifu_class_name) {
        std::vector<cv::Rect> zifu_boxes;
        std::vector<float> zifu_confs;
        std::vector<int> zifu_pos;  // instances_out 中 zifu 实例下标
        for (int i = 0; i < (int)instances_out.size(); ++i) {
            if (instances_out[i].class_name == config_.zifu_class_name) {
                zifu_boxes.push_back(instances_out[i].bbox);
                zifu_confs.push_back(instances_out[i].confidence);
                zifu_pos.push_back(i);
            }
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(zifu_boxes, zifu_confs, 0.f, 0.45f, keep);
        std::vector<char> keep_mask(instances_out.size(), 0);
        for (int k : keep) keep_mask[zifu_pos[k]] = 1;
        std::vector<GxJingzhengSegInstance> filtered;
        for (int i = 0; i < (int)instances_out.size(); ++i) {
            if (instances_out[i].class_name != config_.zifu_class_name || keep_mask[i]) {
                filtered.push_back(std::move(instances_out[i]));
            }
        }
        instances_out = std::move(filtered);
    }

    return branch;
}

int GxJingzhengPipeline::classifyDirection(const std::vector<cv::Mat>& char_images,
                                            int* cls_out, float* conf_out)
{
    if (char_images.empty()) return 0;
    // 方向分类置信度阈值（对称处理，低置信度一律反向）：
    //   判为翻转(180)且平均置信度 < 阈值 -> 不翻转(0)
    //   判为正向(0)且平均置信度  < 阈值 -> 反向翻转(180)
    constexpr float kFlipConfThreshold = 0.95f;
    int count_0 = 0, count_180 = 0;
    float conf_0_sum = 0.f;     // 投"0/正向"票的置信度之和
    float conf_180_sum = 0.f;   // 投"180/翻转"票的置信度之和
    for (auto& img : char_images) {
        if (img.empty()) continue;
        cv::Mat bgr;
        if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        else bgr = img;

        std::vector<cv::Mat> imgs = {bgr};
        std::vector<ClassificationResult> results;
        direction_cls_->process(imgs, results);
        accumulateTiming(t_cls_, direction_cls_->lastBatchTiming());
        if (!results.empty()) {
            if (results[0].class_id == 0) { count_0++; conf_0_sum += results[0].confidence; }
            else { count_180++; conf_180_sum += results[0].confidence; }
        }
    }
    // 多数票为"翻转(180)"时，平均置信度 >= 阈值才翻转
    if (count_180 > count_0) {
        float avg_conf = conf_180_sum / count_180;
        if (cls_out) *cls_out = 180;
        if (conf_out) *conf_out = avg_conf;
        return (avg_conf >= kFlipConfThreshold) ? 180 : 0;
    }
    // 多数票为"正向(0)"时，平均置信度 < 阈值则反向翻转(180)
    if (count_0 > count_180) {
        float avg_conf = conf_0_sum / count_0;
        if (cls_out) *cls_out = 0;
        if (conf_out) *conf_out = avg_conf;
        return (avg_conf < kFlipConfThreshold) ? 180 : 0;
    }
    if (cls_out) *cls_out = -1;
    if (conf_out) *conf_out = 0.f;
    return 0;
}

bool GxJingzhengPipeline::handleZifuBranch(const cv::Mat& crop,
                                            const std::vector<GxJingzhengSegInstance>& zifu_instances,
                                            GxJingzhengPipelineResult& result,
                                            bool verbose)
{
    if (zifu_instances.empty()) {
        if (verbose) std::cout << "[DEBUG] no zifu instance" << std::endl;
        return false;
    }

    auto runOcr = [&](const cv::Mat& img, std::string& text, float& conf) {
        text.clear();
        conf = 0.f;
        if (img.empty()) return;
        cv::Mat bgr;
        if (img.channels() == 1) cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        else bgr = img;
        std::vector<cv::Mat> ocr_imgs = {bgr};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);
        accumulateTiming(t_ocr_, ocr_->lastBatchTiming());
        OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];
        if (!ocr_res.boxes.empty()) {
            text = ocr_res.boxes[0].text;
            conf = ocr_res.boxes[0].confidence;
        }
    };

    struct Piece {
        cv::RotatedRect rrect;       // crop 坐标系
        cv::Mat image_rect_crop;     // boundingRect 直接截取
        cv::Mat warped_before;       // 仿射变换后，未方向矫正
        cv::Mat warped_after;        // 方向矫正后（实际送 OCR 的图）
        int dir_flag = 0;
        std::string text;
        float center_y = 0.f;        // 用于上→下排序
    };

    std::vector<Piece> pieces;
    pieces.reserve(zifu_instances.size());

    for (const auto& inst : zifu_instances) {
        if (inst.mask.empty()) continue;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(inst.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;
        auto biggest = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        cv::RotatedRect rr = cv::minAreaRect(*biggest);
        if (rr.size.width < 2.f || rr.size.height < 2.f) continue;

        // boundingRect 直接截取（用于可视化）
        cv::Rect br = rr.boundingRect();
        cv::Rect safe_br = br & cv::Rect(0, 0, crop.cols, crop.rows);
        cv::Mat rect_crop;
        if (safe_br.area() > 0) rect_crop = crop(safe_br).clone();

        // 透视变换：把 src 的"长边"映射到 dst 的水平边
        cv::Point2f src_pts[4];
        rr.points(src_pts);

        float w_long = std::max(rr.size.width, rr.size.height);
        float h_short = std::min(rr.size.width, rr.size.height);
        int out_w = std::max(2, static_cast<int>(std::round(w_long)));
        int out_h = std::max(2, static_cast<int>(std::round(h_short)));

        float d01 = static_cast<float>(cv::norm(src_pts[0] - src_pts[1]));
        float d12 = static_cast<float>(cv::norm(src_pts[1] - src_pts[2]));
        cv::Point2f dst_pts[4];
        if (d01 >= d12) {
            dst_pts[0] = cv::Point2f(0,           0);
            dst_pts[1] = cv::Point2f(out_w - 1.f, 0);
            dst_pts[2] = cv::Point2f(out_w - 1.f, out_h - 1.f);
            dst_pts[3] = cv::Point2f(0,           out_h - 1.f);
        } else {
            dst_pts[0] = cv::Point2f(0,           out_h - 1.f);
            dst_pts[1] = cv::Point2f(0,           0);
            dst_pts[2] = cv::Point2f(out_w - 1.f, 0);
            dst_pts[3] = cv::Point2f(out_w - 1.f, out_h - 1.f);
        }
        cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat warped;
        cv::warpPerspective(crop, warped, M, cv::Size(out_w, out_h));
        if (warped.empty()) continue;

        Piece p;
        p.rrect = rr;
        p.image_rect_crop = rect_crop;
        p.warped_before = warped;
        p.center_y = rr.center.y;
        pieces.push_back(std::move(p));
    }

    if (pieces.empty()) {
        if (verbose) std::cout << "[DEBUG] zifu pieces empty after warp" << std::endl;
        return false;
    }

    // 逐片段：方向分类 → 矫正 → OCR；低置信度时反向再试取最优
    bool any_flipped = false;
    for (size_t i = 0; i < pieces.size(); i++) {
        auto& p = pieces[i];

        int dir = classifyDirection({p.warped_before});

        cv::Mat ocr_img;
        if (dir == 180) {
            cv::flip(p.warped_before, ocr_img, -1);
        } else {
            ocr_img = p.warped_before;
        }

        std::string text;
        float conf = 0.f;
        runOcr(ocr_img, text, conf);

        // OCR 置信度 < 0.9：反向再试一次，取置信度更高者
        constexpr float kOcrConfLow = 0.9f;
        if (conf < kOcrConfLow) {
            cv::Mat opposite;
            cv::flip(ocr_img, opposite, -1);
            std::string text2;
            float conf2 = 0.f;
            runOcr(opposite, text2, conf2);
            if (conf2 > conf) {
                text = text2;
                conf = conf2;
                ocr_img = opposite;
                dir = (dir == 0) ? 180 : 0;
            }
        }

        p.dir_flag = dir;
        p.warped_after = ocr_img;
        p.text = text;
        if (dir == 180) any_flipped = true;

        if (verbose) {
            std::cout << "[DEBUG]   zifu piece[" << i << "]"
                      << " text=\"" << p.text << "\""
                      << " flip=" << p.dir_flag
                      << " ocr_conf=" << conf << std::endl;
        }
    }

    // 排序：优先 2/6 开头片段在前，同优先级内降序排列
    std::sort(pieces.begin(), pieces.end(),
        [](const Piece& a, const Piece& b) {
            auto priority = [](const std::string& t) -> int {
                if (t.empty()) return -1;
                if (t[0] == '6') return 2;
                if (t[0] == '2') return 1;
                return 0;
            };
            int pa = priority(a.text), pb = priority(b.text);
            if (pa != pb) return pa > pb;
            return a.text > b.text;
        });

    // 构建结果，多个片段之间用 '#' 分隔
    std::string concatenated;
    result.chars.clear();
    result.chars.reserve(pieces.size());
    for (size_t k = 0; k < pieces.size(); k++) {
        auto& p = pieces[k];
        GxJingzhengCharInfo cinfo;
        cinfo.bbox = p.rrect.boundingRect();
        cinfo.image_rect_crop = p.image_rect_crop.clone();
        cinfo.image_before_flip = p.warped_before.clone();
        cinfo.image_after_flip = p.warped_after.clone();
        cinfo.ocr_text = p.text;
        result.chars.push_back(std::move(cinfo));
        if (k > 0) concatenated += "#";
        concatenated += p.text;
    }

    result.direction_flag = any_flipped ? 180 : 0;
    result.ocr_text = concatenated;
    result.rotated_crop = cv::Mat();
    return !concatenated.empty();
}

cv::Mat GxJingzhengPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const GxJingzhengPipelineResult& result)
{
    cv::Mat annotated = src_img.clone();

    // 第 1 层：所有 det 框（最左被选中的高亮），不写框上方文字标签
    // 颜色统一偏浅：选中框用浅薄荷绿描边；未选中框用浅灰
    const cv::Scalar chosen_edge(180, 255, 200);   // 浅薄荷绿 (BGR)
    const cv::Scalar other_edge(190, 190, 190);    // 浅灰
    for (size_t i = 0; i < result.det_detections.size(); i++) {
        const auto& det = result.det_detections[i];
        bool is_chosen = (det.bbox == result.chosen_bbox);
        cv::Rect r = det.bbox & cv::Rect(0, 0, annotated.cols, annotated.rows);
        if (r.area() <= 0) continue;
        if (is_chosen) {
            cv::rectangle(annotated, det.bbox, chosen_edge, 3, cv::LINE_AA);
        } else {
            cv::rectangle(annotated, det.bbox, other_edge, 2, cv::LINE_AA);
        }
    }

    // 第 2 层：分支标签与最终 OCR (左上角)
    {
        std::string head = "branch=" + (result.branch.empty() ? std::string("?") : result.branch)
                           + " state=" + result.state_flag;
        cv::putText(annotated, head, cv::Point(20, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 255), 4);

        if (result.branch == config_.gangbiao_class_name && !result.ocr_text.empty()) {
            // gangbiao: 每个 tiebiao 结果 (逗号分隔的 类型#炉号) 单独成行、红色大字
            std::vector<std::string> items;
            {
                std::string cur;
                for (char c : result.ocr_text) {
                    if (c == ',') { items.push_back(cur); cur.clear(); }
                    else cur += c;
                }
                if (!cur.empty()) items.push_back(cur);
            }
            int y = 200;
            for (const auto& it : items) {
                if (it.empty()) continue;
                std::string line = "ocr: " + it;
                if (y > annotated.rows) break;
                cv::putText(annotated, line, cv::Point(20, y),
                            cv::FONT_HERSHEY_SIMPLEX, 4.5, cv::Scalar(0, 0, 255), 9);
                y += 170;
            }
        } else if (!result.ocr_text.empty()) {
            // zifu 等其它分支：单行红色
            cv::putText(annotated, "ocr: " + result.ocr_text,
                        cv::Point(20, 200),
                        cv::FONT_HERSHEY_SIMPLEX, 4.0, cv::Scalar(0, 0, 255), 8);
        }
    }

    // zifu 分支：底部拼接三行图像条
    // Row1: image_rect_crop（boundingRect 截取）  Row2: image_before_flip（仿射变换后）  Row3: image_after_flip（方向矫正后）
    if (result.branch == config_.zifu_class_name && !result.chars.empty()) {
        const int row_h = 100;
        const int margin = 20;
        const int char_margin = 10;
        const int row_gap = 10;

        auto buildStrip = [&](const std::vector<cv::Mat>& imgs) -> cv::Mat {
            if (imgs.empty()) return cv::Mat();
            std::vector<cv::Mat> resized;
            int total_w = 0;
            for (auto& img : imgs) {
                if (img.empty()) continue;
                float s = static_cast<float>(row_h) / std::max(img.rows, 1);
                cv::Mat r;
                cv::resize(img, r, cv::Size(), s, s, cv::INTER_LINEAR);
                if (r.channels() == 1) cv::cvtColor(r, r, cv::COLOR_GRAY2BGR);
                resized.push_back(r);
                total_w += r.cols + char_margin;
            }
            if (resized.empty()) return cv::Mat();
            total_w = std::max(total_w - char_margin, 1);
            total_w = std::min(total_w, annotated.cols - 2 * margin);

            cv::Mat strip(row_h, total_w, CV_8UC3, cv::Scalar(0, 0, 0));
            int cx = 0;
            for (auto& r : resized) {
                if (cx + r.cols > strip.cols) break;
                r.copyTo(strip(cv::Rect(cx, 0, r.cols, r.rows)));
                cx += r.cols + char_margin;
            }
            return strip;
        };

        std::vector<cv::Mat> imgs_rect, imgs_before, imgs_after;
        for (auto& ch : result.chars) {
            if (!ch.image_rect_crop.empty()) imgs_rect.push_back(ch.image_rect_crop);
            if (!ch.image_before_flip.empty()) imgs_before.push_back(ch.image_before_flip);
            if (!ch.image_after_flip.empty()) imgs_after.push_back(ch.image_after_flip);
        }

        cv::Mat strip1 = buildStrip(imgs_rect);
        cv::Mat strip2 = buildStrip(imgs_before);
        cv::Mat strip3 = buildStrip(imgs_after);

        // 计算总高度：每个 strip 高度 = row_h，间隙 = row_gap
        int used_h = 0;
        if (!strip1.empty()) used_h += row_h + row_gap;
        if (!strip2.empty()) used_h += row_h + row_gap;
        if (!strip3.empty()) used_h += row_h;
        if (used_h == 0) {
            return annotated;
        }
        int extra_h = used_h + 2 * margin;
        cv::Mat canvas = cv::Mat::zeros(annotated.rows + extra_h, annotated.cols, annotated.type());
        annotated.copyTo(canvas(cv::Rect(0, 0, annotated.cols, annotated.rows)));

        int y_pos = annotated.rows + margin;
        if (!strip1.empty()) {
            int paste_w = std::min(strip1.cols, canvas.cols - 2 * margin);
            strip1(cv::Rect(0, 0, paste_w, strip1.rows))
                .copyTo(canvas(cv::Rect(margin, y_pos, paste_w, strip1.rows)));
            y_pos += row_h + row_gap;
        }
        if (!strip2.empty()) {
            int paste_w = std::min(strip2.cols, canvas.cols - 2 * margin);
            strip2(cv::Rect(0, 0, paste_w, strip2.rows))
                .copyTo(canvas(cv::Rect(margin, y_pos, paste_w, strip2.rows)));
            y_pos += row_h + row_gap;
        }
        if (!strip3.empty()) {
            int paste_w = std::min(strip3.cols, canvas.cols - 2 * margin);
            strip3(cv::Rect(0, 0, paste_w, strip3.rows))
                .copyTo(canvas(cv::Rect(margin, y_pos, paste_w, strip3.rows)));
        }
        annotated = canvas;
    }

    // gangbiao 分支：底部追加 tiebiao 标注图的下方信息条 (缩略图/字符片段)，缩放到原图宽度
    if (result.branch == config_.gangbiao_class_name &&
        !result.tiebiao_annotated.empty() && result.chosen_bbox.area() > 0) {
        int crop_h = result.chosen_bbox.height;
        if (result.tiebiao_annotated.rows > crop_h &&
            result.tiebiao_annotated.cols > 0) {
            cv::Mat strip = result.tiebiao_annotated(
                cv::Rect(0, crop_h, result.tiebiao_annotated.cols,
                         result.tiebiao_annotated.rows - crop_h)).clone();
            cv::Mat dimmed;
            strip.convertTo(dimmed, -1, 0.6, 20);   // 与框内贴回一致的整体压暗
            float s = static_cast<float>(annotated.cols) / std::max(dimmed.cols, 1);
            cv::Mat strip_r;
            cv::resize(dimmed, strip_r, cv::Size(), s, s, cv::INTER_LINEAR);

            const int margin = 20;
            int extra_h = strip_r.rows + 2 * margin;
            cv::Mat canvas = cv::Mat::zeros(annotated.rows + extra_h, annotated.cols, annotated.type());
            annotated.copyTo(canvas(cv::Rect(0, 0, annotated.cols, annotated.rows)));
            strip_r.copyTo(canvas(cv::Rect(0, annotated.rows + margin, strip_r.cols, strip_r.rows)));
            annotated = canvas;
        }
    }

    return annotated;
}

void GxJingzhengPipeline::fillTiming(GxJingzhengPipelineResult& r, double total_ms)
{
    r.timing.det = t_det_;
    r.timing.seg = t_seg_;
    r.timing.cls = t_cls_;
    r.timing.ocr = t_ocr_;
    r.timing.total_ms = total_ms;
    // device 取首个有值的模型（实际执行设备，cuda 下可能因 EP 失败降级为 cpu）
    if (!t_det_.device.empty()) r.timing.device = t_det_.device;
    else if (!t_seg_.device.empty()) r.timing.device = t_seg_.device;
    else if (!t_cls_.device.empty()) r.timing.device = t_cls_.device;
    else if (!t_ocr_.device.empty()) r.timing.device = t_ocr_.device;
    else r.timing.device = config_.device;
}

GxJingzhengPipelineResult GxJingzhengPipeline::process(const cv::Mat& image,
                                                        int station_id,
                                                        const std::string& heat_number,
                                                        bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    GxJingzhengPipelineResult result;
    result.state_flag = "NG";
    result.penma_version = "new";

    // 复位各模型耗时累加器
    t_det_ = t_seg_ = t_cls_ = t_ocr_ = InferenceTiming{};

    // ===== 1) 定位检测 =====
    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    accumulateTiming(t_det_, det_->lastBatchTiming());
    DetectionResult det_res = det_results.empty() ? DetectionResult{} : det_results[0];
    result.det_detections = det_res.detections;
    // duanmian: 第一阶段 det 有输出即 yes
    result.duanmian = (det_res.num_detections > 0) ? "yes" : "no";

    if (verbose) {
        std::cout << "[DEBUG] dingwei det = " << det_res.num_detections << std::endl;
    }
    if (det_res.num_detections <= 0) {
        fillTiming(result, std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::high_resolution_clock::now() - t0).count());
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
        fillTiming(result, std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::high_resolution_clock::now() - t0).count());
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

    // ===== 3) 实例分割 → 决定分支 =====
    std::vector<GxJingzhengSegInstance> instances;
    std::string branch = decideBranch(crop, instances);
    if (verbose) {
        std::cout << "[DEBUG] instance seg = " << instances.size()
                  << " masks, branch = \"" << branch << "\"" << std::endl;
        for (size_t i = 0; i < instances.size(); i++) {
            std::cout << "[DEBUG]   inst[" << i << "] class=\"" << instances[i].class_name
                      << "\" mask_pixels=" << cv::countNonZero(instances[i].mask) << std::endl;
        }
    }
    result.branch = branch;
    result.seg_instances = instances;
    if (branch == config_.zifu_class_name) {
        result.zifu_type = "Penma";
    } else if (branch == config_.gangbiao_class_name) {
        result.zifu_type = "Tiebiao";
    }

    if (branch == config_.zifu_class_name) {
        // ===== 4a) zifu 分支：每个实例独立 minAreaRect → 透视裁剪 → 方向分类 + OCR =====
        std::vector<GxJingzhengSegInstance> zifu_insts;
        for (auto& inst : result.seg_instances) {
            if (inst.class_name == config_.zifu_class_name && !inst.mask.empty()) {
                zifu_insts.push_back(inst);
            }
        }
        if (handleZifuBranch(crop, zifu_insts, result, verbose)) {
            result.state_flag = "OK";
        }
    } else if (branch == config_.gangbiao_class_name) {
        // ===== 4b) gangbiao 分支：复用 gx 已切好的 gangbiao bbox 作为 label ROI，转给 tiebiao =====
        if (!tiebiao_pipeline_) {
            std::cerr << "[ERROR] gangbiao branch but tiebiao_pipeline_ not initialized" << std::endl;
        } else {
            std::vector<cv::Rect> gangbiao_bboxes;
            for (auto& inst : result.seg_instances) {
                if (inst.class_name == config_.gangbiao_class_name) {
                    gangbiao_bboxes.push_back(inst.bbox);   // crop 坐标系，与 tiebiao 收到的 image 一致
                }
            }
            if (verbose) std::cout << "[DEBUG] gangbiao bbox reused by tiebiao: "
                                   << gangbiao_bboxes.size() << std::endl;
            TiebiaoResult tres = tiebiao_pipeline_->process(
                crop, station_id, heat_number, gangbiao_bboxes, verbose);
            result.ocr_text = tres.ocr_text;
            result.state_flag = tres.state_flag;
            result.tiebiao_annotated = tres.annotated_image;
        }
    }

    result.annotated_image = createAnnotatedImage(image, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fillTiming(result, static_cast<double>(ms));
    if (verbose) {
        std::cout << "[DEBUG] gx_jingzheng timing summary (device=" << result.timing.device << "):" << std::endl;
        auto printRow = [&](const char* name, const InferenceTiming& t) {
            std::cout << "[DEBUG]   " << name << " : n=" << t.count
                      << " prep=" << t.preprocess_ms << "ms";
            if (t.h2d_split) {
                std::cout << " h2d=" << t.h2d_ms << "ms"
                          << " infer=" << (t.run_ms - t.h2d_ms) << "ms";
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
        std::cout << "[DEBUG]   total: " << result.timing.total_ms << " ms" << std::endl;
    }
    return result;
}

} // namespace Pipeline
} // namespace JHDeepCore
