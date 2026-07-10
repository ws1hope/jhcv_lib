#include "zbhc_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

namespace {

// 计算字符串 a、b 的公共前缀长度
size_t commonPrefixLen(const std::string& a, const std::string& b)
{
    size_t m = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < m && a[i] == b[i]) i++;
    return i;
}

// 给定各片段文本（按 y 顺序），依据 heat_number 找出一种排列，
// 使拼接结果与 heat_number 的公共前缀最长（让 heat_number 尽量成为结果开头）。
// heat_number 为空 / 片段过多(>8) / 无更优解时，退化为原始 y 顺序。
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

// 片段对应炉号的哪个部位
enum HeatTarget : int { kHead = 0, kTail = 1 };

// 逐位匹配计数（按位置字符相等计数，非公共前缀）：head=`266067`、t=`2X6067` -> 5
int matchCount(const std::string& a, const std::string& b)
{
    int n = static_cast<int>(std::min(a.size(), b.size()));
    int c = 0;
    for (int i = 0; i < n; i++) if (a[i] == b[i]) c++;
    return c;
}

struct HeatMatch { HeatTarget target; int score; int max; };

// 判定片段更像 head(前6) 还是 tail(后2，作为片段前缀)：
// 取两者匹配数较大者；tail 为空时只算 head。
HeatMatch matchSegmentToHeat(const std::string& text,
                              const std::string& head, const std::string& tail)
{
    int sh = matchCount(head, text);
    int st = tail.empty() ? -1 : matchCount(tail, text);
    if (sh >= st) return {kHead, sh, static_cast<int>(head.size())};
    return {kTail, st, static_cast<int>(tail.size())};
}

// 用炉号部位覆盖片段文本：
//   HEAD -> 整段替换为 head（6位）
//   TAIL -> 前 tail.size() 位替换为 tail，保留其余后缀为 OCR 结果
std::string overrideWithHeat(const std::string& text, const HeatMatch& m,
                              const std::string& head, const std::string& tail)
{
    if (m.target == kHead) return head;
    if (static_cast<int>(text.size()) > static_cast<int>(tail.size()))
        return tail + text.substr(tail.size());
    return tail;
}

} // namespace

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

ZbhcPipelineResult ZbhcPipeline::process(const cv::Mat& image, bool verbose,
                                         const std::string& heat_number)
{
    auto infer_start = std::chrono::high_resolution_clock::now();

    ZbhcPipelineResult result;

    // 炉号拆分：前 6 位 head + 后 2 位 tail。仅当炉号 >=8 位时启用炉号路径
    // （用炉号匹配定朝向 + 覆盖纠错，替代不准的方向分类器）；否则不做朝向处理。
    const bool use_heat = (heat_number.size() >= 8);
    const std::string head = use_heat ? heat_number.substr(0, 6) : std::string{};
    const std::string tail = use_heat ? heat_number.substr(6) : std::string{};
    if (verbose && use_heat) {
        std::cout << "[DEBUG] heat split: head=\"" << head
             << "\" tail=\"" << tail << "\"" << std::endl;
    }

    // 单图 OCR 封装：输出识别文本与置信度
    auto runOcr = [&](const cv::Mat& img, std::string& text, float& conf) {
        text.clear();
        conf = 0.f;
        if (img.empty()) return;
        std::vector<cv::Mat> ocr_imgs = {img};
        std::vector<OCRResult> ocr_results;
        ocr_->process(ocr_imgs, ocr_results);
        OCRResult ocr_res = ocr_results.empty() ? OCRResult{} : ocr_results[0];
        if (!ocr_res.boxes.empty()) {
            text = ocr_res.boxes[0].text;
            conf = ocr_res.boxes[0].confidence;
        }
    };

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
            billet_result.billet_image = billet_img.clone();

            // ===== Step 5: 实例分割 =====
            std::vector<cv::Mat> seg_imgs = {billet_img};
            std::vector<InstanceSegmentationResult> seg_results;
            seg_->process(seg_imgs, seg_results);
            InstanceSegmentationResult seg_res = seg_results.empty() ? InstanceSegmentationResult{} : seg_results[0];

            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] seg instances: "
                     << seg_res.num_detections << std::endl;
            }

            // 按 y 坐标排序分割实例（自上而下送入 OCR）
            std::vector<int> seg_indices(seg_res.num_detections);
            std::iota(seg_indices.begin(), seg_indices.end(), 0);
            std::sort(seg_indices.begin(), seg_indices.end(), [&](int a, int b) {
                return seg_res.detections[a].bbox.y < seg_res.detections[b].bbox.y;
            });

            // ===== Step 6: 遍历分割实例，最小外接矩算倾角并水平化，收集字符裁剪框 =====
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

                // 从 mask 提取轮廓，取最大轮廓计算外接矩形与倾角
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(char_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (contours.empty()) continue;

                int largest_idx = 0;
                double max_area = 0.0;
                for (int k = 0; k < (int)contours.size(); k++) {
                    double a = cv::contourArea(contours[k]);
                    if (a > max_area) { max_area = a; largest_idx = k; }
                }
                const std::vector<cv::Point>& largest = contours[largest_idx];

                cv::Rect char_bbox_local = cv::boundingRect(largest);
                char_bbox_local = char_bbox_local & cv::Rect(0, 0, billet_img.cols, billet_img.rows);
                if (char_bbox_local.area() <= 0) continue;

                // 最小外接矩，得到字符倾角（minAreaRect.angle ∈ [-90,0)）
                cv::RotatedRect min_rect = cv::minAreaRect(largest);
                float angle = min_rect.angle;
                if (min_rect.size.width < min_rect.size.height) {
                    angle += 90.0f;   // 竖立矩归一到水平方向
                }
                angle = -angle;       // 水平化：旋转抵消倾角

                // 裁剪字符图像、mask 并转 BGR
                cv::Mat char_img = billet_img(char_bbox_local).clone();
                cv::Mat char_mask_crop = char_mask(char_bbox_local).clone();
                if (char_img.empty()) continue;
                cv::Mat char_img_bgr;
                if (char_img.channels() == 1) {
                    cv::cvtColor(char_img, char_img_bgr, cv::COLOR_GRAY2BGR);
                } else {
                    char_img_bgr = char_img;
                }

                // 水平化：加边距后绕字符中心旋转，避免旋转时裁切字符
                // pad 取 W+H（>= 对角线，任意角度都不裁切），旋转后再紧裁回去
                int W = char_bbox_local.width, H = char_bbox_local.height;
                int pad = W + H + 2;
                cv::Mat padded_img, padded_mask;
                cv::copyMakeBorder(char_img_bgr, padded_img, pad, pad, pad, pad,
                                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                cv::copyMakeBorder(char_mask_crop, padded_mask, pad, pad, pad, pad,
                                   cv::BORDER_CONSTANT, cv::Scalar(0));
                cv::Point2f rot_center(
                    min_rect.center.x - char_bbox_local.x + pad,
                    min_rect.center.y - char_bbox_local.y + pad);
                cv::Mat rotM = cv::getRotationMatrix2D(rot_center, angle, 1.0);
                cv::Mat leveled_img, leveled_mask;
                cv::warpAffine(padded_img, leveled_img, rotM, padded_img.size(),
                               cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                cv::warpAffine(padded_mask, leveled_mask, rotM, padded_mask.size(),
                               cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

                // 旋转后重新紧裁：用旋转后的 mask 取轴对齐外接框去掉 padding，
                // 否则 padding 会被方向分类/OCR/可视化当作字符的一部分，导致字符被等比缩小
                cv::Rect tight = cv::boundingRect(leveled_mask);
                int m = 2;
                tight.x = std::max(0, tight.x - m);
                tight.y = std::max(0, tight.y - m);
                tight.width = std::min(leveled_img.cols - tight.x, tight.width + 2 * m);
                tight.height = std::min(leveled_img.rows - tight.y, tight.height + 2 * m);
                if (tight.area() <= 0) continue;
                cv::Mat leveled = leveled_img(tight).clone();

                CharCrop crop;
                crop.bbox_local = char_bbox_local;
                crop.mask = char_mask;
                crop.img_bgr = leveled;   // 水平化并紧裁后的字符（供方向分类与 OCR）
                crop.confidence = char_det.confidence;
                char_crops.push_back(std::move(crop));
            }

            // ===== Step 7+8: 逐片段用炉号匹配定朝向(0/180) + 覆盖纠错，再送 OCR =====
            // 炉号路径：原向 OCR+匹配 head/tail；不达阈值则翻转再 OCR+匹配；
            //   命中阈值(HEAD>=4, TAIL=tail.size()) -> 用炉号部位覆盖文本。
            //   两方向均不达标 -> 取匹配数较高者(并列选原向)，不覆盖。
            // 无炉号路径：不做朝向处理，原向 OCR。
            bool any_flipped = false;
            for (auto& c : char_crops) {
                int char_dir = 0;
                std::string char_text;
                float char_conf = 0.f;
                bool overridden = false;
                int target = -1, score = 0, score_max = 0, thr = 0;
                cv::Mat ocr_in = c.img_bgr;   // 最终采纳朝向的图（供 after_flip 可视化）

                if (use_heat) {
                    // 原向 OCR
                    std::string t0; float conf0 = 0.f;
                    runOcr(c.img_bgr, t0, conf0);
                    HeatMatch m0 = matchSegmentToHeat(t0, head, tail);
                    int thr0 = (m0.target == kHead) ? 4 : static_cast<int>(tail.size());

                    if (m0.score >= thr0 && thr0 > 0) {
                        char_dir = 0;
                        char_text = overrideWithHeat(t0, m0, head, tail);
                        char_conf = conf0; overridden = true;
                        target = m0.target; score = m0.score; score_max = m0.max; thr = thr0;
                    } else {
                        // 翻转 180° 再 OCR
                        cv::Mat flipped;
                        cv::flip(c.img_bgr, flipped, -1);
                        std::string t1; float conf1 = 0.f;
                        runOcr(flipped, t1, conf1);
                        HeatMatch m1 = matchSegmentToHeat(t1, head, tail);
                        int thr1 = (m1.target == kHead) ? 4 : static_cast<int>(tail.size());

                        if (m1.score >= thr1 && thr1 > 0) {
                            char_dir = 180;
                            char_text = overrideWithHeat(t1, m1, head, tail);
                            char_conf = conf1; overridden = true;
                            ocr_in = flipped; any_flipped = true;
                            target = m1.target; score = m1.score; score_max = m1.max; thr = thr1;
                        } else {
                            // 两方向均不达标：取匹配数较高者（并列选原向）
                            if (m1.score > m0.score) {
                                char_dir = 180; char_text = t1; char_conf = conf1;
                                ocr_in = flipped; any_flipped = true;
                                target = m1.target; score = m1.score; score_max = m1.max; thr = thr1;
                            } else {
                                char_dir = 0; char_text = t0; char_conf = conf0;
                                target = m0.target; score = m0.score; score_max = m0.max; thr = thr0;
                            }
                        }
                    }
                } else {
                    // 无炉号：不做朝向处理
                    runOcr(c.img_bgr, char_text, char_conf);
                }

                if (verbose) {
                    const char* tgt = (target == kHead) ? "HEAD"
                                    : (target == kTail) ? "TAIL" : "-";
                    std::cout << "[DEBUG]     char text=\"" << char_text
                         << "\" target=" << tgt
                         << " match=" << score << "/" << score_max
                         << " thr=" << thr
                         << " flip=" << char_dir
                         << (overridden ? " override=Y" : " override=N")
                         << " seg=" << c.confidence
                         << " ocr=" << char_conf << std::endl;
                }

                BilletCharInfo char_info;
                char_info.bbox_on_billet = c.bbox_local;
                char_info.bbox_on_src = cv::Rect(
                    billet_roi_on_src.x + c.bbox_local.x,
                    billet_roi_on_src.y + c.bbox_local.y,
                    c.bbox_local.width,
                    c.bbox_local.height);
                char_info.mask = c.mask;
                char_info.image_before_flip = c.img_bgr.clone();
                char_info.image_after_flip = ocr_in.clone();
                char_info.ocr_text = char_text;
                char_info.ocr_confidence = char_conf;
                char_info.direction_flag = char_dir;
                billet_result.chars.push_back(char_info);
            }

            billet_result.direction_flag = any_flipped ? 180 : 0;
            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] direction: " << billet_result.direction_flag
                     << " (per-char, any flipped)" << std::endl;
            }

            // 依据 heat_number 重排字符片段顺序，使拼接结果前缀尽量匹配炉号；
            // heat_number 为空时 bestOrderByHeat 退化为原始 y 顺序
            std::vector<std::string> seg_texts;
            seg_texts.reserve(billet_result.chars.size());
            std::string yorder;
            for (auto& ch : billet_result.chars) {
                seg_texts.push_back(ch.ocr_text);
                yorder += ch.ocr_text;
            }
            std::vector<int> best_order = bestOrderByHeat(seg_texts, heat_number);

            std::vector<BilletCharInfo> reordered;
            reordered.reserve(best_order.size());
            std::string billet_ocr;
            for (int idx : best_order) {
                reordered.push_back(std::move(billet_result.chars[idx]));
                billet_ocr += seg_texts[idx];
            }
            billet_result.chars = std::move(reordered);
            billet_result.ocr_text = billet_ocr;

            if (verbose && !heat_number.empty() && billet_ocr != yorder) {
                std::cout << "[DEBUG]   billet[" << bi << "] y-order=\"" << yorder
                     << "\" -> heat-ordered=\"" << billet_ocr
                     << "\" (heat=\"" << heat_number << "\")" << std::endl;
            }

            // 坯料置信度 = 各已识别字符(ocr_text 非空)置信度的最小值
            float bmin = 1.0f;
            bool has_rec = false;
            for (auto& ch : billet_result.chars) {
                if (!ch.ocr_text.empty()) {
                    bmin = std::min(bmin, ch.ocr_confidence);
                    has_rec = true;
                }
            }
            billet_result.ocr_confidence = has_rec ? bmin : 0.0f;

            if (verbose) {
                std::cout << "[DEBUG]   billet[" << bi << "] ocr_text=\"" << billet_ocr
                     << "\" ocr_conf=" << billet_result.ocr_confidence << std::endl;
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
            // 字符框右上角标注角度分类结果，右下角标注 OCR 置信度
            std::string dir_label = "d" + std::to_string(ch.direction_flag);
            cv::putText(annotated, dir_label,
                        cv::Point(ch.bbox_on_src.x + ch.bbox_on_src.width + 3,
                                  ch.bbox_on_src.y + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
            std::string conf_label = cv::format("%.2f", ch.ocr_confidence);
            cv::putText(annotated, conf_label,
                        cv::Point(ch.bbox_on_src.x + ch.bbox_on_src.width + 3,
                                  ch.bbox_on_src.y + ch.bbox_on_src.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
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

    // ===== 底部可视化：每个坯料 = 一张卡片（缩略图 + 字符片段），卡片间横向排列、自动换行 =====
    // 参考 gx_jingzheng tiebiao branch 的可视化方式
    const int margin = 15;
    const int row_h = 60;
    const int thumb_height = 200;
    const int char_margin = 4;
    const int gap = 15;
    const int section_gap = 10;
    const int pad_bottom = 15;
    const double strip_font_scale = 0.7;

    // 把多张字符图缩放到统一高度 row_h，并按最大宽度自动折行拼成多行
    // 返回每行 strip（宽度可能不同，高度均为 row_h），供调用方逐行粘贴
    auto buildWrappedStrips = [&](const std::vector<cv::Mat>& imgs, int max_w) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> resized;
        for (auto& img : imgs) {
            if (img.empty()) continue;
            float s = static_cast<float>(row_h) / img.rows;
            cv::Mat r;
            cv::resize(img, r, cv::Size(), s, s, cv::INTER_LINEAR);
            resized.push_back(r);
        }
        std::vector<cv::Mat> lines;
        if (resized.empty()) return lines;
        std::vector<cv::Mat> cur; int cur_w = 0;
        auto flush = [&]() {
            if (cur.empty()) return;
            int total_w = 0;
            for (auto& r : cur) total_w += r.cols + char_margin;
            total_w -= char_margin;
            cv::Mat line = cv::Mat::zeros(row_h, total_w, CV_8UC3);
            int cx = 0;
            for (auto& r : cur) {
                cv::Rect roi(cx, 0, r.cols, r.rows);
                r.copyTo(line(roi));
                cx += r.cols + char_margin;
            }
            lines.push_back(line);
            cur.clear(); cur_w = 0;
        };
        for (auto& r : resized) {
            int add = r.cols + (cur.empty() ? 0 : char_margin);
            if (!cur.empty() && cur_w + add > max_w) flush();
            cur_w += add;
            cur.push_back(r);
        }
        flush();
        return lines;
    };

    const int label_h = 20;            // 每行 strip 上方文字标签高度
    const int line_pitch = row_h + label_h;  // 单行字符（含标签）总高

    // 每个坯料打包成一个卡片：缩略图（左）+ 字符 strip 多行（右，自动折行）
    struct Card {
        cv::Mat thumb;                 // 缩略图（已缩放到 thumb_height 高）
        int thumb_w = 0;
        std::vector<cv::Mat> strip_lines;  // 字符 strip 各行（0°：1+ 行；180°：before 段 + after 段）
        int before_lines = 0;          // 180° 时 before 段行数，用于粘贴时区分标签
        int strip_w = 0;               // strip 区域宽度（取各行最大宽度）
        int strip_h = 0;               // strip 区域高度（含标签）
        int card_w = 0, card_h = 0;    // 整卡尺寸
    };

    int avail_w = annotated.cols - 2 * margin;   // 卡片可用的水平空间
    std::vector<Card> cards;
    cards.reserve(billets.size());

    for (auto& billet : billets) {
        Card c;
        if (!billet.billet_image.empty()) {
            float thumb_scale = static_cast<float>(thumb_height) / billet.billet_image.rows;
            cv::resize(billet.billet_image, c.thumb, cv::Size(), thumb_scale, thumb_scale, cv::INTER_LINEAR);
            c.thumb_w = c.thumb.cols;
        }
        int strip_max_w = std::max(1, avail_w - c.thumb_w - gap);

        std::vector<cv::Mat> imgs_before, imgs_after;
        for (auto& ch : billet.chars) {
            imgs_before.push_back(ch.image_before_flip);
            imgs_after.push_back(ch.image_after_flip);
        }

        if (billet.direction_flag == 180) {
            auto before = buildWrappedStrips(imgs_before, strip_max_w);
            auto after = buildWrappedStrips(imgs_after, strip_max_w);
            c.before_lines = (int)before.size();
            // before 段
            for (auto& l : before) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)before.size() * line_pitch;
            // after 段（带额外段间距）
            if (!after.empty()) c.strip_h += section_gap;
            for (auto& l : after) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)after.size() * line_pitch;
        } else {
            auto lines = buildWrappedStrips(imgs_after, strip_max_w);
            for (auto& l : lines) { c.strip_lines.push_back(l); c.strip_w = std::max(c.strip_w, l.cols); }
            c.strip_h += (int)lines.size() * line_pitch;
        }

        c.card_w = c.thumb_w + (c.strip_lines.empty() ? 0 : gap + c.strip_w);
        c.card_h = std::max(thumb_height, c.strip_h);
        cards.push_back(std::move(c));
    }

    // 无有效卡片时不扩展画布
    bool any_content = false;
    for (auto& c : cards) if (c.card_w > 0) { any_content = true; break; }
    if (!any_content) {
        return annotated;
    }

    // 第一遍：流式排版（卡片间横向排列、放不下换行），只算总高度
    int cur_x = margin, cur_y = 0, row_max_h = 0, total_h = 0;
    for (auto& c : cards) {
        if (c.card_w <= 0) continue;
        if (cur_x + c.card_w > annotated.cols - margin && cur_x > margin) {
            cur_y += row_max_h + section_gap;
            cur_x = margin;
            row_max_h = 0;
        }
        cur_x += c.card_w + gap;
        row_max_h = std::max(row_max_h, c.card_h);
        total_h = std::max(total_h, cur_y + row_max_h);
    }
    total_h += pad_bottom;

    // 扩展画布：原图标注置顶，下方追加可视化区域
    int canvas_h = annotated.rows + total_h;
    cv::Mat result = cv::Mat::zeros(canvas_h, annotated.cols, annotated.type());
    annotated.copyTo(result(cv::Rect(0, 0, annotated.cols, annotated.rows)));

    // 带越界裁剪的粘贴（保证绝不出界）
    auto pasteAt = [&](const cv::Mat& img, int x, int y) {
        if (img.empty()) return;
        cv::Rect paste_rect(x, y, img.cols, img.rows);
        cv::Rect clipped = paste_rect & cv::Rect(0, 0, result.cols, result.rows);
        if (clipped.empty()) return;
        cv::Rect src_roi(clipped.x - x, clipped.y - y, clipped.width, clipped.height);
        img(src_roi).copyTo(result(clipped));
    };

    // 第二遍：在确定画布上重新流式排版并粘贴
    cur_x = margin;
    int row_top = annotated.rows;
    row_max_h = 0;
    for (int bi = 0; bi < (int)cards.size(); bi++) {
        auto& c = cards[bi];
        if (c.card_w <= 0) continue;
        std::string tag = "billet" + std::to_string(bi + 1);

        // 放不下则换行
        if (cur_x + c.card_w > result.cols - margin && cur_x > margin) {
            row_top += row_max_h + section_gap;
            cur_x = margin;
            row_max_h = 0;
        }

        int card_top = row_top;
        int thumb_x = cur_x;
        int strip_x = cur_x + c.thumb_w + (c.strip_lines.empty() ? 0 : gap);

        // 缩略图竖直居中于卡片（标签由各 strip 行的 "billetN ..." 文字标识，不另画）
        if (!c.thumb.empty()) {
            int thumb_y = card_top + (c.card_h - c.thumb.rows) / 2;
            pasteAt(c.thumb, thumb_x, thumb_y);
        }

        // 字符 strip 逐行粘贴
        int ly = card_top;
        for (int li = 0; li < (int)c.strip_lines.size(); li++) {
            // 180° 的 after 段首行前补段间距，与 strip_h 计算一致
            if (billets[bi].direction_flag == 180 && li == c.before_lines && c.before_lines > 0) {
                ly += section_gap;
            }
            std::string lbl;
            if (billets[bi].direction_flag == 180) {
                lbl = (li < c.before_lines) ? (tag + " dir=180 Before flip:")
                                            : (tag + " dir=180 After flip:");
            } else {
                lbl = tag + " Char crops:";
            }
            cv::putText(result, lbl, cv::Point(strip_x, ly + label_h - 5),
                        cv::FONT_HERSHEY_SIMPLEX, strip_font_scale, cv::Scalar(200, 200, 200), 2);
            pasteAt(c.strip_lines[li], strip_x, ly + label_h);
            ly += line_pitch;
        }

        cur_x += c.card_w + gap;
        row_max_h = std::max(row_max_h, c.card_h);
    }

    return result;
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

} // namespace Pipeline
} // namespace JHDeepCore
