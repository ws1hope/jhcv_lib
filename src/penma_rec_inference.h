#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolo_inference.h"
#include "ocr_inference.h"
#include "cls_inference.h"
#include "cvHelp.h"

struct PenmaRecParams {
    std::string label_model_path;
    std::string zifu_model_path;
    std::string ocr_rec_model_path;
    std::string ocr_rec_label_path;
    std::string cls_model_path;
    bool useGPU = true;
    int gpuId = 0;
    float label_conf_threshold = 0.25f;
    float zifu_conf_threshold = 0.25f;
    float label_nms_threshold = 0.45f;
    float zifu_nms_threshold = 0.45f;
    int yolo_input_width = 640;
    int yolo_input_height = 640;
    int cls_input_width = 48;
    int cls_input_height = 192;
};

struct PenmaLabelInfo {
    int class_id;
    cv::Rect bbox;
    cv::Mat roi_image;
};

struct PenmaCharInfo {
    int center_x;
    int center_y;
    std::string text;
    int cls_label;
    cv::Mat char_image;
};

struct PenmaRecResult {
    bool success = false;
    std::string ocr_result;
    cv::Mat annotated_image;
    cv::Mat rotated_image;
    std::vector<PenmaLabelInfo> labels;
    std::vector<PenmaCharInfo> characters;
};

class PenmaRecInference {
public:
    explicit PenmaRecInference(const PenmaRecParams& params);
    ~PenmaRecInference();

    void warmup();
    PenmaRecResult recognize(const cv::Mat& image, const std::string& heat_str = "");
    PenmaRecResult recognize(const std::string& image_path, const std::string& heat_str = "");

private:
    std::vector<PenmaLabelInfo> detectLabels(const cv::Mat& image);
    std::vector<OcrAngleParams> detectChars(const cv::Mat& label_roi, cv::Mat& rotated_image);
    std::vector<PenmaCharInfo> recognizeChars(const std::vector<cv::Mat>& char_images);
    
    cv::Mat createMaskFromContour(const cv::Rect& bbox, const std::vector<cv::Point>& contour);
    void determineOrientation(std::vector<PenmaCharInfo>& chars, 
                              const std::vector<OcrAngleParams>& angle_params,
                              int& direction_flag);
    std::string postprocessResult(std::vector<PenmaCharInfo>& chars, 
                                  const std::string& heat_str,
                                  int label_class_id);
    
    cv::Rect safeROI(int x, int y, int w, int h, int img_w, int img_h);

    PenmaRecParams m_params;
    std::unique_ptr<YOLOInference> m_label_detector;
    std::unique_ptr<YOLOInference> m_zifu_detector;
    std::unique_ptr<OCRInference> m_ocr;
    std::unique_ptr<ResNetInference> m_cls;
};

int countCommonChars(const std::string& str1, const std::string& str2);
bool isFirstNumLargest(float a, float b, float c);
bool isFirstNumSmallest(float a, float b, float c);