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
} // namespace JHDeepCore

class InferHelper {
public:
    InferHelper() = delete;

    static cv::Rect safeROI(int x, int y, int w, int h, int img_w, int img_h);

    static std::string sortCharsByPosition(
        const std::vector<JHDeepCore::Detection>& char_dets,
        const std::vector<std::string>& char_texts);
};
