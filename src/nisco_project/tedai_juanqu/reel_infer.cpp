#include "reel_infer.h"

#include <iostream>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// ---------- ReelDetector ----------

ReelDetector::ReelDetector(const std::string &model_path, bool use_gpu,
                           float conf_threshold, float iou_threshold)
{
    try {
        // ai_platform 约定：device_id >= 0 -> cuda，< 0 -> cpu
        detector_ = std::make_unique<Detector>(model_path, "", use_gpu ? 0 : -1,
                                               "", conf_threshold, iou_threshold);
        valid_ = true;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel detector init failed (" << model_path
                  << "): " << e.what() << std::endl;
        detector_.reset();
        valid_ = false;
    }
}

bool ReelDetector::detect(const cv::Mat &frame, std::vector<ReelDetObject> &objects)
{
    objects.clear();
    if (frame.empty() || !valid_) return false;

    std::vector<cv::Mat> images{frame};
    std::vector<DetectionResult> results;
    try {
        detector_->process(images, results);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel detector inference failed: " << e.what()
                  << std::endl;
        return false;
    }
    if (results.empty()) return false;

    for (const auto &det : results[0].detections) {
        ReelDetObject obj;
        obj.classid = det.class_id;
        obj.prob = det.confidence;
        // ai_platform 输出左上角 bbox -> 转中心点（业务层中心点语义不变）
        obj.rect.x = det.bbox.x + det.bbox.width / 2;
        obj.rect.y = det.bbox.y + det.bbox.height / 2;
        obj.rect.width = det.bbox.width;
        obj.rect.height = det.bbox.height;
        objects.push_back(obj);
    }
    return true;
}

// ---------- ReelClassifier ----------

ReelClassifier::ReelClassifier(const std::string &model_path, bool use_gpu)
{
    try {
        classifier_ = std::make_unique<Classifier>(model_path, "", use_gpu ? 0 : -1);
        valid_ = true;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel classifier init failed (" << model_path
                  << "): " << e.what() << std::endl;
        classifier_.reset();
        valid_ = false;
    }
}

ReelClassifier::Result ReelClassifier::detect(const cv::Mat &frame)
{
    Result res;
    if (frame.empty() || !valid_) return res;

    std::vector<cv::Mat> images{frame};
    std::vector<ClassificationResult> results;
    try {
        classifier_->process(images, results);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel classifier inference failed: " << e.what()
                  << std::endl;
        return res;
    }
    if (results.empty()) return res;

    res.valid = true;
    res.class_id = results[0].class_id;
    res.confidence = results[0].confidence;
    return res;
}

} // namespace Pipeline
} // namespace JHDeepCore
