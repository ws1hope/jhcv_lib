#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "JHDeepCore.h"

namespace JHDeepCore {
struct DabangJiguangResult {
    int picture_id = 0;
    std::string state_flag;
    std::string zifu_type;
    std::string ocr_text;
    std::string picture_path;
};

struct TiebiaoResult {
    int picture_id = 0;
    std::string state_flag;
    std::string label_type;
    std::string ocr_text;
    std::string picture_path;
    cv::Mat annotated_image;
};

struct GxJingzhengResult {
    int picture_id = 0;
    std::string state_flag;        // "OK" / "NG"
    std::string branch;            // "zifu" / "gangbiao" / ""
    std::string ocr_text;          // 识别结果
    std::string picture_path;
    cv::Mat annotated_image;
};
} // namespace JHDeepCore

class InferHelper {
public:
    InferHelper() = delete;

    static cv::Rect safeROI(int x, int y, int w, int h, int img_w, int img_h);

    static std::string sortCharsByPosition(
        const std::vector<JHDeepCore::Detection>& char_dets,
        const std::vector<std::string>& char_texts);

    static std::string fixLuhaoChars(std::string text);

    static int findBestLuhaoMatch(const std::vector<std::string>& ocr_texts,
                                   const std::string& heat_number);
};
