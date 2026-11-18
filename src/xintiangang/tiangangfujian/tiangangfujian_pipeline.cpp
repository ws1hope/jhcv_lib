#include "tiangangfujian_pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace JHDeepCore {
namespace Pipeline {

// 各断面 OCR 文本用逗号连接（多断面时 result 形如 "xxx,yyy"）
static std::string joinSectionTexts(const std::vector<TiangangfujianSectionResult>& sections)
{
    std::string out;
    for (const auto& sec : sections) {
        if (sec.ocr_text.empty()) continue;
        if (!out.empty()) out += ",";
        out += sec.ocr_text;
    }
    return out;
}

// penma 编号纠错（config_.char_correction 开启时调用）：
//   绝对第 3 位固定 'T'；绝对第 11 位限定 A/W/X，识别为其他字符（如 K）改为 'X'
// offset：本段文本在整段结果中的起始绝对位置（0 基，按字符拆分时传入前序字符总长）；
// 文本长度不足的位跳过。
static std::string correctPenmaTextAt(const std::string& in, int offset)
{
    std::string out = in;
    // 绝对位 pos（1 基）对应 out 下标 pos - 1 - offset
    int idx3 = 3 - 1 - offset;
    if (idx3 >= 0 && idx3 < (int)out.size() && out[idx3] != 'T') {
        out[idx3] = 'T';
    }
    int idx11 = 11 - 1 - offset;
    if (idx11 >= 0 && idx11 < (int)out.size()) {
        char& c = out[idx11];
        if (c != 'A' && c != 'W' && c != 'X') c = 'X';
    }
    return out;
}

TiangangfujianPipeline::TiangangfujianPipeline(const TiangangfujianServerConfig& config)
    : config_(config)
{
    int dev_id = (config_.device == "cuda" || config_.device == "gpu") ? 0 : -1;

    det_ = std::make_unique<Detector>(config_.det_model, "", dev_id);
    std::cout << "[OK] Tiangangfujian section det model loaded: " << config_.det_model << std::endl;

    // iou=0.9：与 tiebiao 的 char_seg_ 一致，宽松保留全部字符候选框，
    // 避免密集字符在 NMS 阶段被误去重。
    char_seg_ = std::make_unique<InstanceSegmenter>(config_.char_seg_model, "", dev_id, "",
                                                     0.25f, 0.9f);
    std::cout << "[OK] Tiangangfujian char seg model loaded: " << config_.char_seg_model << std::endl;

    ocr_ = std::make_unique<OCRRecognizer>(config_.ocr_model, config_.ocr_label, dev_id);
    std::cout << "[OK] Tiangangfujian OCR model loaded: " << config_.ocr_model << std::endl;

    warmup();
}

void TiangangfujianPipeline::warmup()
{
    std::cout << "[INFO] Warming up tiangangfujian models..." << std::endl;
    std::vector<cv::Mat> dummy = {cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};

    std::vector<DetectionResult> det_results;
    det_->process(dummy, det_results);
    std::cout << "[OK] Section detector warmed up" << std::endl;

    std::vector<InstanceSegmentationResult> seg_results;
    char_seg_->process(dummy, seg_results);
    std::cout << "[OK] Char seg warmed up" << std::endl;

    std::vector<OCRResult> ocr_results;
    ocr_->process(dummy, ocr_results);
    std::cout << "[OK] OCR recognizer warmed up" << std::endl;

    std::cout << "[OK] Warmup complete." << std::endl;
}

TiangangfujianPipelineResult TiangangfujianPipeline::process(const cv::Mat& image, bool verbose)
{
    TiangangfujianPipelineResult result;
    std::vector<TiangangfujianSectionResult> sections;

    if (config_.rois.empty()) {
        // 未配置 ROI：整图作为一个区域处理
        if (verbose) std::cout << "[DEBUG] No rois configured, process whole image" << std::endl;
        processRegion(image, cv::Point(0, 0), 0, verbose, sections);
    } else {
        // 全局固定 ROI 列表：逐区域裁剪后独立处理
        if (verbose) std::cout << "[DEBUG] " << config_.rois.size() << " roi(s) configured" << std::endl;
        for (int i = 0; i < (int)config_.rois.size(); i++) {
            const auto& r = config_.rois[i];
            cv::Rect roi = ImageHelper::safeClampROI(
                r[0], r[1], r[2], r[3], image.cols, image.rows);
            if (roi.area() <= 0) {
                if (verbose) std::cout << "[DEBUG] roi" << (i + 1) << " invalid, skip" << std::endl;
                continue;
            }
            if (verbose) {
                std::cout << "[DEBUG] roi" << (i + 1) << " = (" << roi.x << "," << roi.y
                          << "," << roi.width << "," << roi.height << ")" << std::endl;
            }
            cv::Mat region = image(roi).clone();
            processRegion(region, roi.tl(), i + 1, verbose, sections);
        }
    }

    // 断面按原图 x 从左到右排序，拼接结果符合阅读顺序
    std::sort(sections.begin(), sections.end(),
        [](const TiangangfujianSectionResult& a, const TiangangfujianSectionResult& b) {
            return a.bbox_on_src.x < b.bbox_on_src.x;
        });

    result.sections = sections;
    result.all_results = joinSectionTexts(sections);
    result.annotated_image = createAnnotatedImage(image, result.sections);

    return result;
}

void TiangangfujianPipeline::processRegion(const cv::Mat& region_img,
                                            const cv::Point& region_offset,
                                            int roi_index,
                                            bool verbose,
                                            std::vector<TiangangfujianSectionResult>& sections)
{
    std::vector<cv::Mat> det_imgs = {region_img};
    std::vector<DetectionResult> det_results;
    det_->process(det_imgs, det_results);
    DetectionResult det_result = det_results.empty() ? DetectionResult{} : det_results[0];

    if (verbose) {
        std::cout << "[DEBUG] region roi" << roi_index
                  << " sections detected: " << det_result.num_detections << std::endl;
    }

    std::vector<Detection> dets = det_result.detections;
    // 区域内断面按 x 从左到右
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.bbox.x < b.bbox.x;
        });

    for (int i = 0; i < (int)dets.size(); i++) {
        const auto& d = dets[i];

        if (verbose) {
            std::cout << "  sec[" << i << "] class=" << d.class_name
                 << " conf=" << d.confidence
                 << " bbox=(" << d.bbox.x << "," << d.bbox.y
                 << "," << d.bbox.width << "," << d.bbox.height
                 << ")" << std::endl;
        }

        cv::Rect roi = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            region_img.cols, region_img.rows);
        if (roi.area() <= 0) continue;

        TiangangfujianSectionResult sec;
        sec.roi_index = roi_index;
        sec.class_name = d.class_name;
        sec.confidence = d.confidence;
        sec.class_id = d.class_id;
        sec.bbox_on_src = cv::Rect(roi.tl() + region_offset, roi.br() + region_offset);

        cv::Mat section_img = region_img(roi).clone();
        if (section_img.empty()) continue;
        sec.section_image = section_img;

        // 字符实例分割 + 断面角度矫正
        cv::Mat rotated_image;
        std::vector<CharCropInfo> char_crops;
        detectAndRotateChars(section_img, rotated_image, char_crops);
        sec.rotated_image = rotated_image;

        if (char_crops.empty()) {
            if (verbose) std::cout << "  sec[" << i << "] no chars detected" << std::endl;
            sections.push_back(std::move(sec));
            continue;
        }

        if (verbose) std::cout << "  sec[" << i << "] "
                               << char_crops.size() << " chars detected" << std::endl;

        // 按矫正后 y 坐标从小到大排序（从上到下阅读顺序）
        std::sort(char_crops.begin(), char_crops.end(),
            [](const CharCropInfo& a, const CharCropInfo& b) {
                return a.center_y < b.center_y;
            });

        // 方向判别（几何规则，暂不用模型）：按定位框最长边关系判断 0/180 度
        std::vector<cv::Mat> char_images_before;
        for (auto& crop : char_crops) {
            char_images_before.push_back(crop.image.clone());
        }
        int dir_flag = classifyDirectionByGeometry(char_crops);
        sec.direction_flag = dir_flag;
        if (verbose) std::cout << "  sec[" << i << "] direction: " << dir_flag
                               << (dir_flag == 180 ? " (flip 180)" : "") << std::endl;

        for (int ci = 0; ci < (int)char_crops.size(); ci++) {
            if (dir_flag == 180 && !char_crops[ci].image.empty()) {
                cv::Mat flipped;
                cv::flip(char_crops[ci].image, flipped, -1);
                char_crops[ci].image = flipped.clone();
            }
        }

        if (dir_flag == 180) {
            // 180 度时整个断面上下颠倒，字符阅读顺序同步反转
            // （例：502/070832X/2672 -> 26T2/070832X/502）；
            // char_images_before 与 char_crops 一一对应，需同步反转
            std::reverse(char_crops.begin(), char_crops.end());
            std::reverse(char_images_before.begin(), char_images_before.end());
        }

        // 逐字符 OCR，按排序顺序拼接
        std::string text;
        int text_offset = 0;
        for (int ci = 0; ci < (int)char_crops.size(); ci++) {
            TiangangfujianCharResult chr;
            chr.bbox_on_rotated = char_crops[ci].bbox;
            chr.image_before_flip = char_images_before[ci];
            chr.image_after_flip = char_crops[ci].image;
            chr.direction_flag = dir_flag;

            cv::Mat bgr;
            if (char_crops[ci].image.channels() == 1)
                cv::cvtColor(char_crops[ci].image, bgr, cv::COLOR_GRAY2BGR);
            else
                bgr = char_crops[ci].image;

            std::vector<cv::Mat> ocr_imgs = {bgr};
            std::vector<OCRResult> ocr_results;
            ocr_->process(ocr_imgs, ocr_results);

            std::string t;
            if (!ocr_results.empty() && !ocr_results[0].boxes.empty()) {
                t = ocr_results[0].boxes[0].text;
            }
            chr.ocr_text = config_.char_correction
                               ? correctPenmaTextAt(t, text_offset) : t;
            if (verbose) {
                std::cout << "    char[" << ci << "] text=\"" << t << "\""
                          << (config_.char_correction
                                  ? " -> \"" + chr.ocr_text + "\"" : "")
                          << std::endl;
            }
            sec.chars.push_back(std::move(chr));

            text += t;
            text_offset += (int)t.size();
        }

        sec.ocr_text = config_.char_correction
                           ? correctPenmaTextAt(text, 0) : text;
        sections.push_back(std::move(sec));
    }
}

void TiangangfujianPipeline::detectAndRotateChars(const cv::Mat& section_roi,
                                                   cv::Mat& rotated_image,
                                                   std::vector<CharCropInfo>& char_crops)
{
    char_crops.clear();

    std::vector<cv::Mat> imgs = {section_roi};
    std::vector<InstanceSegmentationResult> results;
    char_seg_->process(imgs, results);

    if (results.empty() || results[0].num_detections <= 0) return;

    auto& det = results[0];
    std::vector<CharAngleInfo> char_infos;

    for (int i = 0; i < det.num_detections; i++) {
        auto& d = det.detections[i];
        auto& mask_full = det.masks[i];

        cv::Rect safe = ImageHelper::safeClampROI(
            d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height,
            section_roi.cols, section_roi.rows);
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

    if (char_infos.empty()) return;

    // 取高度最大字符的角度作为整断面矫正角度
    ImageHelper::sortByHeight(char_infos);
    float angle = char_infos.back().angle;

    int sum_x = 0, sum_y = 0;
    for (auto& info : char_infos) {
        sum_x += info.center_x + info.bbox_offset.x;
        sum_y += info.center_y + info.bbox_offset.y;
    }
    int center_x = sum_x / static_cast<int>(char_infos.size());
    int center_y = sum_y / static_cast<int>(char_infos.size());

    // bias 边距：旋转后角点可能越出原断面范围，四周补 30 像素黑边
    const int bias = 30;
    cv::Mat padded = cv::Mat::zeros(
        section_roi.rows + 2 * bias, section_roi.cols + 2 * bias, CV_8UC3);
    section_roi.copyTo(padded(cv::Rect(bias, bias, section_roi.cols, section_roi.rows)));

    int x_off = 0, y_off = 0;
    rotated_image = ImageHelper::rotateImageAroundPoint(
        padded, center_x + bias, center_y + bias, angle, x_off, y_off);

    char_crops = ImageHelper::rotateAndRemapBBoxes(
        char_infos, cv::Point2f(static_cast<float>(center_x),
                                 static_cast<float>(center_y)),
        angle, rotated_image, bias, bias, x_off, y_off);
}

// 几何规则判向（暂不用模型）：
// 三个字符按定位框 bbox.y 从小到大排列，取各自定位框最长边 max(width, height)：
//   中间字符最长边为三者最大，且 上 > 下 -> 序列「第二长 最长 最短」-> 0 度
//   中间字符最长边为三者最大，且 上 < 下 -> 序列「最短 最长 第二长」-> 180 度
// 其余情况（非 3 个字符 / 中间不是最长 / 上下相等）默认 0 度。
// 在副本上排序，不改变调用方 char_crops 的顺序（OCR 拼接仍按原顺序）。
int TiangangfujianPipeline::classifyDirectionByGeometry(
    const std::vector<CharCropInfo>& char_crops)
{
    if (char_crops.size() != 3) return 0;

    std::vector<CharCropInfo> sorted = char_crops;
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const CharCropInfo& a, const CharCropInfo& b) {
            return a.bbox.y < b.bbox.y;
        });

    int len_top = std::max(sorted[0].bbox.width, sorted[0].bbox.height);
    int len_mid = std::max(sorted[1].bbox.width, sorted[1].bbox.height);
    int len_bot = std::max(sorted[2].bbox.width, sorted[2].bbox.height);

    // 中间字符必须是最长边最大者，否则规则不适用
    if (len_mid <= len_top || len_mid <= len_bot) return 0;

    if (len_top > len_bot) return 0;     // 第二长 最长 最短
    if (len_top < len_bot) return 180;   // 最短 最长 第二长
    return 0;                            // 上下相等无法判断，默认 0 度
}

// 归一为 3 通道 BGR，供结果图拼接（灰度/4通道/3通道通吃；空图返回空）(同 luqian)
static cv::Mat toBgrImage(const cv::Mat& img)
{
    if (img.empty()) return cv::Mat();
    if (img.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (img.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    return img;
}

// 等比缩放到指定高度（宽按比例），空图返回空 (同 luqian)
static cv::Mat scaleToHeight(const cv::Mat& img, int h)
{
    if (img.empty() || h <= 0) return cv::Mat();
    if (img.rows == h) return img.clone();
    int w = std::max(1, static_cast<int>(std::round(img.cols * static_cast<double>(h) / img.rows)));
    cv::Mat out;
    cv::resize(img, out, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    return out;
}

cv::Mat TiangangfujianPipeline::createAnnotatedImage(
    const cv::Mat& src_img,
    const std::vector<TiangangfujianSectionResult>& sections)
{
    cv::Mat annotated = src_img.clone();

    // 层1: 断面 det 框（绿色）+ 标签
    for (int i = 0; i < (int)sections.size(); i++) {
        const auto& s = sections[i];
        cv::rectangle(annotated, s.bbox_on_src, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("sec%d %s %.2f", i + 1,
                                       s.class_name.c_str(), s.confidence);
        cv::putText(annotated, label,
                    cv::Point(s.bbox_on_src.x, s.bbox_on_src.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // 层2: 左上角结果文字（每个断面一行，从上到下，红色大字）；
    // 超宽时右侧加宽画布（同 luqian 层4，但不逗号拼行）
    {
        const double fs = 4.5;      // 字号（加大）
        const int thick = 11;
        const int line_gap = 140;   // 行距
        int max_need_w = 0;
        for (const auto& s : sections) {
            if (s.ocr_text.empty()) continue;
            max_need_w = std::max(max_need_w,
                cv::getTextSize(s.ocr_text, cv::FONT_HERSHEY_SIMPLEX, fs, thick, nullptr).width);
        }
        if (max_need_w + 40 > annotated.cols) {
            cv::copyMakeBorder(annotated, annotated, 0, 0, 0,
                               max_need_w + 40 - annotated.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        int text_y = 120;
        for (const auto& s : sections) {
            if (s.ocr_text.empty()) continue;
            if (text_y <= annotated.rows) {
                cv::putText(annotated, s.ocr_text, cv::Point(20, text_y),
                            cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 0, 255), thick);
            }
            text_y += line_gap;
        }
    }

    // 层3: 底部新增"字符片段方向矫正前/后"对照面板（同 luqian 层5）。
    // 所有断面的字符排在同一对 before/after 行中，每列一个字符并标注断面号。
    // 面板引用各断面字符（列标签 s断面_t序号 "OCR" d方向）
    struct PanelCharRef {
        int sec_index;   // 断面下标（0 基）
        int char_index;  // 断面内字符下标（0 基）
    };
    std::vector<PanelCharRef> refs;
    for (int si = 0; si < (int)sections.size(); si++) {
        for (int ci = 0; ci < (int)sections[si].chars.size(); ci++) {
            refs.push_back({si, ci});
        }
    }

    if (!refs.empty()) {
        const int N = (int)refs.size();
        const int gap = 8;         // 列间距/边距
        const int pad = 4;         // 列内左右留白
        const int label_h = 24;    // 每行缩略图下方标注高度
        const int title_h = 32;    // 顶部标题行高
        const int left_w = 90;     // 左侧 before/after 行标签列宽
        const double label_fs = 0.5;

        // 按参考高度估算各列宽比，用于按字符数自适应缩略图高度
        auto charAt = [&](int k) -> const TiangangfujianCharResult& {
            return sections[refs[k].sec_index].chars[refs[k].char_index];
        };
        std::vector<double> col_aspect(N, 0.0);
        for (int k = 0; k < N; ++k) {
            cv::Mat b = toBgrImage(charAt(k).image_before_flip);
            cv::Mat a = toBgrImage(charAt(k).image_after_flip);
            double ab = b.empty() ? 1.0 : static_cast<double>(b.cols) / b.rows;
            double aa = a.empty() ? 1.0 : static_cast<double>(a.cols) / a.rows;
            col_aspect[k] = std::max(ab, aa);
        }
        double aspect_sum = std::accumulate(col_aspect.begin(), col_aspect.end(), 0.0);
        int target_w = std::max(annotated.cols, 640);
        int avail = target_w - left_w - N * gap - N * 2 * pad;
        int th = (aspect_sum > 0.0)
                     ? static_cast<int>(std::floor(avail / aspect_sum))
                     : 100;
        th = std::max(24, std::min(160, th));   // 字符越多 -> 越小；越少 -> 越大

        // 面板尺寸
        std::vector<int> col_w(N, 0);
        int content_w = 0;
        for (int k = 0; k < N; ++k) {
            col_w[k] = std::max(1, static_cast<int>(std::round(col_aspect[k] * th))) + 2 * pad;
            content_w += col_w[k];
        }
        int panel_w = std::max(annotated.cols, left_w + content_w + (N + 1) * gap);
        int row_h = th + label_h;
        int panel_h = title_h + 2 * row_h + gap;

        // 标题行：右侧追加各断面方向判定结果（几何规则）
        const std::string title = "char segments (before / after direction flip)";
        std::string dir_info = "dir:";
        for (int si = 0; si < (int)sections.size(); si++) {
            dir_info += cv::format(" s%d=%d", si + 1, sections[si].direction_flag);
        }
        dir_info += "  basis: geometry";
        int title_w = cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr).width;
        int dir_w = cv::getTextSize(dir_info, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr).width;
        panel_w = std::max(panel_w, 10 + title_w + 30 + dir_w + 10);

        cv::Mat panel(panel_h, panel_w, CV_8UC3, cv::Scalar(20, 20, 20));

        cv::putText(panel, title, cv::Point(10, title_h - 9),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(panel, dir_info, cv::Point(10 + title_w + 30, title_h - 9),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);

        auto putThumb = [&](const cv::Mat& raw, int col_x, int col_wi, int row_y) {
            cv::Mat bgr = toBgrImage(raw);
            if (bgr.empty()) return;
            cv::Mat s = scaleToHeight(bgr, th);
            if (s.cols > col_wi - 2 * pad) {
                cv::resize(s, s, cv::Size(col_wi - 2 * pad, th));
            }
            int cx = col_x + (col_wi - s.cols) / 2;
            cx = std::max(col_x, std::min(cx, col_x + col_wi - s.cols));
            cv::Rect dst(cx, row_y, s.cols, s.rows);
            dst &= cv::Rect(0, 0, panel.cols, panel.rows);
            if (dst.area() <= 0) return;
            s(cv::Rect(0, 0, dst.width, dst.height)).copyTo(panel(dst));
        };

        const int row0_y = title_h;
        const int row1_y = title_h + row_h;
        // 行标签（Hershey 不支持中文，用英文）
        cv::putText(panel, "before", cv::Point(10, row0_y + th / 2 + 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);
        cv::putText(panel, "after", cv::Point(10, row1_y + th / 2 + 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 1);

        int x = left_w + gap;
        for (int k = 0; k < N; ++k) {
            const TiangangfujianCharResult& ch = charAt(k);
            cv::rectangle(panel, cv::Rect(x, row0_y, col_w[k], th), cv::Scalar(80, 80, 80), 1);
            cv::rectangle(panel, cv::Rect(x, row1_y, col_w[k], th), cv::Scalar(80, 80, 80), 1);
            putThumb(ch.image_before_flip, x, col_w[k], row0_y);
            putThumb(ch.image_after_flip, x, col_w[k], row1_y);
            // 标注：断面号_序号 + OCR 文本 + 方向标志
            std::string lab = cv::format("s%d_t%d \"%s\" d%d",
                refs[k].sec_index + 1, refs[k].char_index + 1,
                ch.ocr_text.c_str(), ch.direction_flag);
            cv::putText(panel, lab, cv::Point(x, row1_y + th + label_h - 7),
                        cv::FONT_HERSHEY_SIMPLEX, label_fs, cv::Scalar(0, 255, 0), 1);
            x += col_w[k] + gap;
        }

        // 拼到结果图底部（宽度取二者较大者，避免超出）
        cv::Mat annotated_bgr = toBgrImage(annotated);
        int out_w = std::max(annotated_bgr.cols, panel_w);
        int out_h = annotated_bgr.rows + panel_h;
        cv::Mat out(out_h, out_w, CV_8UC3, cv::Scalar(0, 0, 0));
        annotated_bgr.copyTo(out(cv::Rect(0, 0, annotated_bgr.cols, annotated_bgr.rows)));
        panel.copyTo(out(cv::Rect(0, annotated_bgr.rows, panel_w, panel_h)));
        annotated = out;
    }

    return annotated;
}

} // namespace Pipeline
} // namespace JHDeepCore
