#include "infer_utils.h"

#include <algorithm>

cv::Rect InferHelper::safeROI(int x, int y, int w, int h, int img_w, int img_h)
{
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, img_w - x);
    h = std::min(h, img_h - y);
    if (w <= 0 || h <= 0) return cv::Rect(0, 0, 0, 0);
    return cv::Rect(x, y, w, h);
}

std::string InferHelper::sortCharsByPosition(
    const std::vector<JHDeepCore::Detection>& char_dets,
    const std::vector<std::string>& char_texts)
{
    if (char_dets.empty()) return "";

    std::vector<std::pair<int, std::string>> chars_with_pos;
    for (int i = 0; i < (int)char_dets.size(); i++) {
        chars_with_pos.emplace_back(char_dets[i].bbox.y, char_texts[i]);
    }

    std::sort(chars_with_pos.begin(), chars_with_pos.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });

    std::string combined;
    bool first = true;
    for (auto& cp : chars_with_pos) {
        combined += cp.second;
        if (first) {
            combined += "#";
            first = false;
        }
    }
    return combined;
}
