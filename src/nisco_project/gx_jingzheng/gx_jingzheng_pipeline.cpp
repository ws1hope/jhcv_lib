#include "gx_jingzheng_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>

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

// 计算字符串 a、b 的公共前缀长度
size_t commonPrefixLen(const std::string& a, const std::string& b)
{
    size_t m = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < m && a[i] == b[i]) i++;
    return i;
}

// 给定各片段文本（按 x 顺序），依据 heat_number 找出一种排列，
// 使拼接结果与 heat_number 的公共前缀最长（让 heat_number 尽量成为结果开头）。
// heat_number 为空 / 片段过多(>8) / 无更优解时，退化为原始 x 顺序。
std::vector<int> bestOrderByHeat(const std::vector<std::string>& texts,
                                  const std::string& heat_number)
{
    int n = static_cast<int>(texts.size());
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 0);

    if (n <= 1 || heat_number.empty() || n > 8) {
        return identity;
    }

    auto concat = [&](const std::vector<int>& order) {
        std::string s;
        for (int i : order) s += texts[i];
        return s;
    };

    std::vector<int> best = identity;
    size_t best_prefix = commonPrefixLen(concat(best), heat_number);

    std::vector<int> cur = identity;
    std::sort(cur.begin(), cur.end());
    do {
        size_t p = commonPrefixLen(concat(cur), heat_number);
        if (p > best_prefix) {
            best_prefix = p;
            best = cur;
        }
    } while (std::next_permutation(cur.begin(), cur.end()));

    return best;
}

// 滑动对齐结果：offset=片段在炉号中的最佳对齐偏移，score=该位置逐位匹配数，overlap=重叠长度
struct AlignResult { int offset; int score; int overlap; };

// 把片段 OCR 文本在整段炉号 heat 上滑动，找逐位匹配数最大的对齐位置。
// offset = heat_index - text_index：text[i] 对齐 heat[i+offset]
//   offset>=0：text[0] 对齐 heat[offset]（片段起点落在炉号内部）
//   offset<0 ：text[-offset] 对齐 heat[0]（片段带炉号前的噪声前缀）
// 片段起点可在任意位置（含跨 head/tail 边界、带噪声前后缀），不要求对齐在炉号首位。
AlignResult bestAlign(const std::string& text, const std::string& heat)
{
    int best_off = 0, best_score = -1, best_ov = 0;
    int Lh = static_cast<int>(heat.size());
    int Lt = static_cast<int>(text.size());
    for (int off = -(Lt - 1); off < Lh; off++) {
        int i0 = std::max(0, -off);
        int i1 = std::min(Lt, Lh - off);
        if (i1 <= i0) continue;
        int c = 0;
        for (int i = i0; i < i1; i++) if (text[i] == heat[i + off]) c++;
        if (c > best_score) {
            best_score = c;
            best_off = off;
            best_ov = i1 - i0;
        }
    }
    return {best_off, std::max(best_score, 0), best_ov};
}

// 滑动对齐命中阈值：匹配数需 >= max(3, 60%*overlap)（强多数，且至少 3 位）
int alignThreshold(int overlap)
{
    return std::max(3, (overlap * 3 + 4) / 5);   // ceil(overlap*0.6)
}

// 用炉号子串覆盖 OCR 文本中与炉号重叠的区域，保留重叠区前后的原始识别结果。
// 仅把 text[i0 .. i0+overlap) 这段替换为 heat 子串；前缀 text[0..i0)、后缀 text[i0+overlap..] 原样保留。
//   i0 = max(0,-offset)：重叠区在 text 中的起点（offset<0 时跳过噪声前缀）。
//   全对(score==overlap)时重叠区内容不变，等价于只保留原始 OCR 的前后缀。
std::string overrideAlign(const std::string& text, const AlignResult& a, const std::string& heat)
{
    int i0 = std::max(0, -a.offset);
    int h_start = std::max(0, a.offset);
    return text.substr(0, i0)
         + heat.substr(h_start, a.overlap)
         + text.substr(i0 + a.overlap);
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
                                            const std::string& heat_number,
                                            bool verbose)
{
    if (zifu_instances.empty()) {
        if (verbose) std::cout << "[DEBUG] no zifu instance" << std::endl;
        return false;
    }

    // 炉号路径：把片段 OCR 在整段炉号上滑动对齐，按最佳对齐位置的逐位匹配数
    // 定朝向(0/180) + 覆盖纠错（替代方向分类器）。仅当炉号 >=8 位时启用；否则原向 OCR。
    const bool use_heat = (heat_number.size() >= 8);
    if (verbose && use_heat) {
        std::cout << "[DEBUG] heat=\"" << heat_number << "\" (sliding align)" << std::endl;
    }

    // 单图 OCR 封装：输出识别文本与置信度
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
        OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];
        if (!ocr_res.boxes.empty()) {
            text = ocr_res.boxes[0].text;
            conf = ocr_res.boxes[0].confidence;
        }
    };

    struct Piece {
        cv::RotatedRect rrect;  // crop 坐标系
        cv::Mat warped_before;  // 透视裁剪后，未方向矫正
        cv::Mat warped_after;   // 方向矫正后（实际送 OCR 的图）
        int dir_flag = 0;
        std::string text;
        float center_x = 0.f;   // 用于左→右排序
    };

    std::vector<Piece> pieces;
    pieces.reserve(zifu_instances.size());

    for (const auto& inst : zifu_instances) {
        if (inst.mask.empty()) continue;

        // 找轮廓 → 取最大者算最小外接矩 (避免噪声小块)
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(inst.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;
        auto biggest = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        cv::RotatedRect rr = cv::minAreaRect(*biggest);
        if (rr.size.width < 2.f || rr.size.height < 2.f) continue;

        // 4 个源点 (crop 坐标系)
        cv::Point2f src_pts[4];
        rr.points(src_pts);

        // 目标矩形：长边 → out_w，短边 → out_h
        float w_long = std::max(rr.size.width, rr.size.height);
        float h_short = std::min(rr.size.width, rr.size.height);
        int out_w = std::max(2, static_cast<int>(std::round(w_long)));
        int out_h = std::max(2, static_cast<int>(std::round(h_short)));

        // rr.points() 顺序：pts[0]=BL, pts[1]=TL, pts[2]=TR, pts[3]=BR
        //   d01 = ||BL-TL|| = rrect.height (左竖边)
        //   d12 = ||TL-TR|| = rrect.width  (上水平边)
        // 目标：把 src 的"长边"映射到 dst 的水平边 (长度 out_w)，
        //       让裁剪出来的图永远是"长边水平、短边竖直"——即旋转到 0°/180° 状态。
        float d01 = static_cast<float>(cv::norm(src_pts[0] - src_pts[1]));
        float d12 = static_cast<float>(cv::norm(src_pts[1] - src_pts[2]));
        cv::Point2f dst_pts[4];
        if (d01 >= d12) {
            // 长边 = BL→TL (竖直方向)，需把图像顺时针旋 90°，使该边变成 dst 上水平边
            //   BL → dst 左上, TL → dst 右上, TR → dst 右下, BR → dst 左下
            dst_pts[0] = cv::Point2f(0,           0);
            dst_pts[1] = cv::Point2f(out_w - 1.f, 0);
            dst_pts[2] = cv::Point2f(out_w - 1.f, out_h - 1.f);
            dst_pts[3] = cv::Point2f(0,           out_h - 1.f);
        } else {
            // 长边 = TL→TR (已水平)，恒等对应
            //   BL → dst 左下, TL → dst 左上, TR → dst 右上, BR → dst 右下
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
        p.warped_before = warped;
        p.center_x = rr.center.x;
        pieces.push_back(std::move(p));
    }

    if (pieces.empty()) {
        if (verbose) std::cout << "[DEBUG] zifu pieces empty after warp" << std::endl;
        return false;
    }

    // 按 x 中心左→右排序（作为初始/兜底顺序）
    std::sort(pieces.begin(), pieces.end(),
        [](const Piece& a, const Piece& b) { return a.center_x < b.center_x; });

    // 第一遍：逐片段用炉号滑动对齐定朝向(0/180) + 覆盖纠错，再送 OCR
    // 炉号路径：原向 OCR+滑动对齐；不达阈值则翻转再 OCR+对齐；
    //   命中阈值(匹配数 >= max(3, 60%*overlap)) -> 用对齐位置的炉号子串覆盖文本。
    //   两方向均不达标 -> 取匹配数较高者(并列选原向)，不覆盖。
    // 无炉号路径：不做朝向处理，原向 OCR。
    bool any_flipped = false;
    for (size_t i = 0; i < pieces.size(); i++) {
        auto& p = pieces[i];
        int char_dir = 0;
        std::string char_text;
        std::string raw_text;          // 采纳朝向的原始 OCR（覆盖前）
        float char_conf = 0.f;
        bool overridden = false;
        int offset = 0, score = 0, overlap = 0, thr = 0;
        cv::Mat ocr_in = p.warped_before;   // 最终采纳朝向的图（供 after_flip 可视化）

        if (use_heat) {
            // 原向 OCR
            std::string t0; float conf0 = 0.f;
            runOcr(p.warped_before, t0, conf0);
            AlignResult a0 = bestAlign(t0, heat_number);
            int thr0 = alignThreshold(a0.overlap);

            if (a0.score >= thr0 && thr0 > 0) {
                char_dir = 0;
                raw_text = t0;
                char_text = overrideAlign(t0, a0, heat_number);
                char_conf = conf0; overridden = true;
                offset = a0.offset; score = a0.score; overlap = a0.overlap; thr = thr0;
            } else {
                // 翻转 180° 再 OCR
                cv::Mat flipped;
                cv::flip(p.warped_before, flipped, -1);
                std::string t1; float conf1 = 0.f;
                runOcr(flipped, t1, conf1);
                AlignResult a1 = bestAlign(t1, heat_number);
                int thr1 = alignThreshold(a1.overlap);

                if (a1.score >= thr1 && thr1 > 0) {
                    char_dir = 180;
                    raw_text = t1;
                    char_text = overrideAlign(t1, a1, heat_number);
                    char_conf = conf1; overridden = true;
                    ocr_in = flipped; any_flipped = true;
                    offset = a1.offset; score = a1.score; overlap = a1.overlap; thr = thr1;
                } else {
                    // 两方向均不达标：取匹配数较高者（并列选原向）
                    if (a1.score > a0.score) {
                        char_dir = 180; raw_text = t1; char_text = t1; char_conf = conf1;
                        ocr_in = flipped; any_flipped = true;
                        offset = a1.offset; score = a1.score; overlap = a1.overlap; thr = thr1;
                    } else {
                        char_dir = 0; raw_text = t0; char_text = t0; char_conf = conf0;
                        offset = a0.offset; score = a0.score; overlap = a0.overlap; thr = thr0;
                    }
                }
            }
        } else {
            // 无炉号：不做朝向处理
            runOcr(p.warped_before, char_text, char_conf);
            raw_text = char_text;
        }

        p.dir_flag = char_dir;
        p.warped_after = ocr_in;
        p.text = char_text;

        if (verbose) {
            std::cout << "[DEBUG]   zifu piece[" << i << "] (x-order)"
                      << " raw=\"" << raw_text << "\""
                      << " text=\"" << p.text << "\""
                      << " offset=" << offset
                      << " match=" << score << "/" << overlap
                      << " thr=" << thr
                      << " flip=" << p.dir_flag
                      << (overridden ? " override=Y" : " override=N")
                      << " ocr=" << char_conf << std::endl;
        }
    }

    // 第二步：根据 heat_number 重新排列各片段顺序，使拼接结果与 heat_number 的公共前缀最长
    std::vector<std::string> texts;
    texts.reserve(pieces.size());
    for (auto& p : pieces) texts.push_back(p.text);

    std::vector<int> best_order = bestOrderByHeat(texts, heat_number);
    if (verbose) {
        std::string xorder;
        for (auto& t : texts) xorder += t;
        std::cout << "[DEBUG] zifu x-order concat=\"" << xorder
                  << "\", heat=\"" << heat_number << "\"" << std::endl;
        std::string reordered;
        for (int idx : best_order) reordered += texts[idx];
        std::cout << "[DEBUG] zifu heat-ordered concat=\"" << reordered << "\"" << std::endl;
    }

    // 按新顺序构建 chars 并拼接最终结果
    // 多个片段之间用 '#' 分隔（即第二个片段起前面加 '#'），单片段则不加分隔符
    std::string concatenated;
    result.chars.clear();
    result.chars.reserve(best_order.size());
    for (size_t k = 0; k < best_order.size(); k++) {
        int idx = best_order[k];
        auto& p = pieces[idx];
        GxJingzhengCharInfo cinfo;
        cinfo.bbox = p.rrect.boundingRect();   // 轴对齐 bbox (仅作信息)
        cinfo.image_before_flip = p.warped_before.clone();
        cinfo.image_after_flip = p.warped_after.clone();
        cinfo.ocr_text = p.text;
        result.chars.push_back(std::move(cinfo));
        if (k > 0) concatenated += "#";
        concatenated += p.text;
    }

    // 整体 direction_flag：任一片段翻转即记 180（与 zbhc 一致，仅用于可视化）
    result.direction_flag = any_flipped ? 180 : 0;
    result.ocr_text = concatenated;
    // 每个 piece 都独立矫正了，整张 rotated_crop 概念已不存在
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

    // zifu 分支：底部拼一个矫正后图 + 字符片段长条
    // zifu 分支：底部拼接所有 piece 的 OCR 输入图（每个 piece 独立矫正后的字符串图）
    if (result.branch == config_.zifu_class_name && !result.chars.empty()) {
        const int row_h = 140;
        const int margin = 20;
        const int char_margin = 12;

        std::vector<cv::Mat> imgs_for_strip;
        for (auto& ch : result.chars) {
            if (!ch.image_after_flip.empty()) imgs_for_strip.push_back(ch.image_after_flip);
        }
        if (imgs_for_strip.empty()) {
            return annotated;
        }

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

        int strip_w = std::min(total_w, annotated.cols - 2 * margin);
        cv::Mat strip = cv::Mat::zeros(row_h, total_w, CV_8UC3);
        int cx = 0;
        for (auto& r : resized) {
            if (cx + r.cols > strip.cols) break;
            r.copyTo(strip(cv::Rect(cx, 0, r.cols, r.rows)));
            cx += r.cols + char_margin;
        }

        int extra_h = row_h + 2 * margin;
        cv::Mat canvas = cv::Mat::zeros(annotated.rows + extra_h, annotated.cols, annotated.type());
        annotated.copyTo(canvas(cv::Rect(0, 0, annotated.cols, annotated.rows)));

        int paste_w = std::min(strip.cols, canvas.cols - 2 * margin);
        if (paste_w > 0) {
            strip(cv::Rect(0, 0, paste_w, strip.rows))
                .copyTo(canvas(cv::Rect(margin, annotated.rows + margin, paste_w, strip.rows)));
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

GxJingzhengPipelineResult GxJingzhengPipeline::process(const cv::Mat& image,
                                                        int station_id,
                                                        const std::string& heat_number,
                                                        bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    GxJingzhengPipelineResult result;
    result.state_flag = "NG";
    result.penma_version = "new";

    // ===== 1) 定位检测 =====
    std::vector<cv::Mat> det_imgs = {image};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_res = det_results.empty() ? DetectionResult{} : det_results[0];
    result.det_detections = det_res.detections;
    // duanmian: 第一阶段 det 有输出即 yes
    result.duanmian = (det_res.num_detections > 0) ? "yes" : "no";

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
        if (handleZifuBranch(crop, zifu_insts, result, heat_number, verbose)) {
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
    if (verbose) {
        std::cout << "[DEBUG] gx_jingzheng total inference time: " << ms << " ms" << std::endl;
    }
    return result;
}

} // namespace Pipeline
} // namespace JHDeepCore
