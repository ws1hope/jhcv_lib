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

int InferHelper::countCommonChars(const std::string& a, const std::string& b)
{
    int count = 0;
    int len = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < len; i++) {
        if (a[i] == b[i]) count++;
    }
    return count;
}

int InferHelper::findBestLuhaoMatch(const std::vector<std::string>& ocr_texts,
                                      const std::string& heat_number)
{
    if (ocr_texts.empty()) return -1;

    std::vector<int> scores(ocr_texts.size(), 0);
    for (int i = 0; i < (int)ocr_texts.size(); i++) {
        scores[i] = countCommonChars(ocr_texts[i], heat_number);
    }

    auto max_it = std::max_element(scores.begin(), scores.end());
    int max_idx = static_cast<int>(std::distance(scores.begin(), max_it));
    int max_val = scores[max_idx];

    if (max_val >= 4) return max_idx;

    for (int i = 0; i < (int)ocr_texts.size(); i++) {
        if (ocr_texts[i].size() > 7 && !ocr_texts[i].empty() &&
            ocr_texts[i][0] >= '0' && ocr_texts[i][0] <= '9') {
            return i;
        }
    }

    return -1;
}
