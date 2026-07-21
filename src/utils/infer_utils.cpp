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

std::string InferHelper::fixLuhaoChars(std::string text)
{
    if (text.empty()) return text;
    text[0] = '2';
    std::replace(text.begin(), text.end(), 'R', '5');
    std::replace(text.begin(), text.end(), 'S', '5');
    if (text.size() > 1) {
        text.back() = (text.back() == 'A') ? '4' : text.back();
    }
    return text;
}

int InferHelper::findBestLuhaoMatch(const std::vector<std::string>& ocr_texts,
                                      const std::string& heat_number)
{
    // 新标准：返回首个“长度 > 9 且前两位为数字、第 1 位 >= '2' 且第 2 位 >= '6'”的结果（例 "27..."）。
    // 不再依据输入炉号 heat_number 做字符匹配；参数保留以兼容现有调用方。
    (void)heat_number;
    if (ocr_texts.empty()) return -1;

    for (int i = 0; i < (int)ocr_texts.size(); i++) {
        const std::string& t = ocr_texts[i];
        if (t.size() > 9 &&
            t[0] >= '2' && t[0] <= '9' &&
            t[1] >= '6' && t[1] <= '9') {
            return i;
        }
    }
    return -1;
}
