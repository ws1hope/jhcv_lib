#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "JHDeepCore.h"
#include "image_utils.h"
#include "file_utils.h"
#include "infer_utils.h"

namespace JHDeepCore {
namespace Pipeline {

class TiebiaoPipeline {
public:
    explicit TiebiaoPipeline(const TiebiaoConfig& config);

    TiebiaoResult process(const cv::Mat& image,
                          int station_id,
                          const std::string& heat_number,
                          bool verbose = false);

private:
    std::vector<std::pair<cv::Mat, int>> detectLabels(const cv::Mat& image);

    void detectAndRotateChars(const cv::Mat& label_roi,
                               cv::Mat& rotated_image,
                               std::vector<CharCropInfo>& char_crops);

    int classifyDirection(const std::vector<cv::Mat>& char_images);

    std::vector<std::string> recognizeChars(const std::vector<CharCropInfo>& crops);

    cv::Mat createAnnotatedImage(const cv::Mat& src_img,
                                  const cv::Mat& rotated_img,
                                  const std::vector<std::string>& ocr_texts,
                                  const std::vector<CharCropInfo>& crops,
                                  const std::string& label_type,
                                  int direction_flag);

    void warmup();

    std::unique_ptr<InstanceSegmenter> label_seg_;
    std::unique_ptr<InstanceSegmenter> char_seg_;
    std::unique_ptr<OCRRecognizer> ocr_;
    std::unique_ptr<Classifier> direction_cls_;
    TiebiaoConfig config_;
};

} // namespace Pipeline
} // namespace JHDeepCore
