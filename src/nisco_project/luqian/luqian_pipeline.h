#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include "file_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

struct LuqianCharInfo {
    cv::Rect bbox_on_target;
    cv::Rect bbox_on_src;
    cv::Mat mask;
    cv::Mat image_before_flip;   // 方向矫正前（原始字符 crop）
    cv::Mat image_after_flip;    // 方向矫正后（实际送 OCR 的字符 crop）
    std::string ocr_text;
    float ocr_confidence = 0.f;  // 该字符 OCR 置信度（无 box 时为 0）
    int direction_flag = 0;     // 该字符角度分类结果（0 或 180）
};

struct LuqianTargetResult {
    cv::Rect bbox_on_src;
    std::string class_name;
    float confidence;
    int direction_flag = 0;   // 0 或 180
    cv::Mat target_image;     // 目标裁剪图（用于结果图可视化缩略图）
    std::vector<LuqianCharInfo> chars;
    std::string ocr_text;
    std::string ocr1_text;   // 第一次识别(模型1)完整拼接结果
    std::string ocr2_text;   // 第二次识别(模型2)完整拼接结果（触发二次识别时填充）
    float ocr_confidence = 0.f;  // 该目标各已识别字符 OCR 置信度的最小值
};

// 各模型分段耗时汇总（毫秒）。device 反映实际执行设备（cuda/cpu）。
// split 时输入在 GPU、run_ms 已排除 H2D/D2H（仍含 ORT host 开销+kernel，非纯 kernel），infer=run_ms；非 split 时 run_ms 含 ORT 内部 H2D/D2H。tensor_ms 预期≈0。
struct LuqianTiming {
    InferenceTiming det;      // 目标检测
    InferenceTiming seg;     // 实例分割
    InferenceTiming cls;    // 方向分类
    InferenceTiming ocr;    // OCR（模型1）
    InferenceTiming ocr2;   // OCR（模型2，二次识别，未配置则为空）
    double total_ms = 0.0;  // process() 总耗时（含非模型部分）
    std::string device;     // 实际设备 cuda/cpu
};

struct LuqianPipelineResult {
    std::vector<Detection> det_detections;
    std::vector<LuqianTargetResult> targets;
    cv::Mat annotated_image;
    LuqianTiming timing;     // 各模型分段耗时 + 总耗时 + 设备
};

class LuqianPipeline {
public:
    explicit LuqianPipeline(const LuqianServerConfig& config);

    LuqianPipelineResult process(const cv::Mat& image, bool verbose = false,
                                 const std::string& heat_number = "");

private:
    void warmup();

    // 方向分类器(pm_fx_cls)逐字符投票定目标朝向；cls_out=-1 表示无结论(票数平局/空)，调用方回退几何法
    int classifyDirection(const std::vector<cv::Mat>& char_images,
                          int* cls_out, float* conf_out);

    cv::Mat createAnnotatedImage(
        const cv::Mat& src_img,
        const std::vector<LuqianTargetResult>& targets);

    // 把本次 process() 累计的各模型耗时写进 result.timing
    void fillTiming(LuqianPipelineResult& r, double total_ms);

    std::unique_ptr<Detector> det_;
    std::unique_ptr<InstanceSegmenter> seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::unique_ptr<OCRRecognizer> ocr2_;   // 二次识别 OCR（可选，为空不启用）
    std::unique_ptr<Classifier> direction_cls_;
    LuqianServerConfig config_;

    // 各模型分段耗时累加器：process() 起点复位，各模型调用后累加
    InferenceTiming t_det_, t_seg_, t_cls_, t_ocr_, t_ocr2_;
};

} // namespace Pipeline
} // namespace JHDeepCore
