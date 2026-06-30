#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "JHDeepCore.h"
#include "file_utils.h"
#include "image_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

/// 单根坯料的完整识别结果
struct ZbsltjBilletResult {
    cv::Rect bbox;                      // 坯料在原图的位置
    cv::Mat rotated_image;              // 字符旋转矫正后的整图
    std::vector<CharCropInfo> char_crops;  // 旋转矫正后的每个字符片段
    std::vector<std::string> rec_texts; // 每片段 OCR 文本（已按方向排序）
    std::string matched_heat;           // 匹配到的炉号
    std::string pdi_count;              // PDI 支数
    int seq_number = 0;                 // 跟踪序号
    bool success = false;
};

class ZbsltjPipeline {
public:
    explicit ZbsltjPipeline(const ZbsltjConfig& config);

    /// 单图处理：返回该图所有坯料的识别结果
    /// current_heat / seq_counter 由调用方持有，跨调用维持状态
    std::vector<ZbsltjBilletResult> process(
        const cv::Mat& image,
        const std::vector<std::string>& candidate_heats,
        const std::vector<std::string>& candidate_pdis,
        std::string& current_heat,    // in/out: 当前计数炉号
        int& seq_counter,             // in/out: 当前序号
        bool verbose = false);

    /// 在原图上绘制识别结果（绿框、文本、序号、炉号、PDI）
    void drawResults(cv::Mat& image,
                     const std::vector<ZbsltjBilletResult>& results) const;

    /// 喷码格式后处理：6#2#3 / 6#3#2 / 旋转 / 6#5
    static std::string formatPenmaResult(const std::string& concat);

    /// 比较两个字符串前 8 位是否相同
    static bool matchFirstEight(const std::string& a, const std::string& b);

private:
    /// 步骤 1：坯料检测
    std::vector<std::pair<cv::Mat, cv::Rect>> detectBillets(const cv::Mat& image);

    /// 步骤 2：字符分割 + 旋转矫正
    bool segmentAndRotateChars(const cv::Mat& billet_roi,
                                cv::Mat& rotated_image,
                                std::vector<CharCropInfo>& char_crops);

    /// 步骤 3：OCR 识别（每个字符片段）
    void recognizeFragments(const std::vector<CharCropInfo>& crops,
                            std::vector<std::string>& texts);

    /// 步骤 4：按 y 排序（自上而下）
    void sortByY(std::vector<CharCropInfo>& crops,
                 std::vector<std::string>& texts) const;

    /// 字符覆盖率相似度（与原项目 calculateSimilarity 等价）
    double calcSimilarity(const std::vector<std::string>& fragments,
                          const std::string& target) const;

    void warmup();

    std::unique_ptr<Detector> billet_det_;
    std::unique_ptr<InstanceSegmenter> char_seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    ZbsltjConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
