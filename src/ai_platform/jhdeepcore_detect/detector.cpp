#include "JHDeepCore.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {

class DetectorPrivate {
public:
    DetectorPrivate(const std::string &model_path, const std::string &label_path,
                    int device_id, const std::string &config_path,
                    float conf_threshold, float iou_threshold)
    {
        std::string device_str = device_id >= 0 ? "cuda" : "cpu";
        std::vector<std::string> names;
        if (!label_path.empty()) names.push_back(label_path);
        inference_ = inference::InferenceFactory::CreateInference(model_path, device_str, names);
        inference_->SetThresholds(conf_threshold, iou_threshold);
    }

    void process(std::vector<cv::Mat> &images, std::vector<DetectionResult> &results) {
        if (!inference_) throw std::runtime_error("Detector not initialized");
        if (images.empty()) {
            results.clear();
            return;
        }
        auto *onnx = dynamic_cast<inference::OnnxInference *>(inference_.get());
        if (!onnx) throw std::runtime_error("Unsupported inference type");
        results = onnx->InferBatchDetection(images);
    }

    size_t get_batch() const {
        if (!inference_) throw std::runtime_error("Detector not initialized");
        return static_cast<size_t>(inference_->GetConfig().batch_size);
    }

    size_t get_input_width() const {
        if (!inference_) throw std::runtime_error("Detector not initialized");
        auto cfg = inference_->GetConfig();
        return static_cast<size_t>(cfg.img_scale.width > 0 ? cfg.img_scale.width : 640);
    }

    size_t get_input_height() const {
        if (!inference_) throw std::runtime_error("Detector not initialized");
        auto cfg = inference_->GetConfig();
        return static_cast<size_t>(cfg.img_scale.height > 0 ? cfg.img_scale.height : 640);
    }

private:
    std::unique_ptr<inference::BaseInference> inference_;
};

Detector::Detector(const std::string &model_path, const std::string &label_path,
                   int device_id, const std::string &config_path,
                   float conf_threshold, float iou_threshold)
    : m_pHandle(std::make_shared<DetectorPrivate>(model_path, label_path, device_id, config_path,
                                                   conf_threshold, iou_threshold)) {}

Detector::~Detector() = default;

void Detector::process(std::vector<cv::Mat> &images, std::vector<DetectionResult> &results) {
    m_pHandle->process(images, results);
}

size_t Detector::GetBatch() const { return m_pHandle->get_batch(); }
size_t Detector::GetInputWidth() const { return m_pHandle->get_input_width(); }
size_t Detector::GetInputHeight() const { return m_pHandle->get_input_height(); }

} // namespace JHDeepCore
