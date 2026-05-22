#include "JHDeepCore.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {

class SegmenterPrivate {
public:
    SegmenterPrivate(const std::string &model_path, const std::string &label_path,
                     int device_id, const std::string &config_path)
    {
        std::string device_str = device_id >= 0 ? "cuda" : "cpu";
        std::vector<std::string> names;
        if (!label_path.empty()) names.push_back(label_path);
        inference_ = inference::InferenceFactory::CreateInference(model_path, device_str, names);
    }

    void process(std::vector<cv::Mat> &images, std::vector<SegmentationResult> &results) {
        if (!inference_) throw std::runtime_error("Segmenter not initialized");
        if (images.empty()) {
            results.clear();
            return;
        }
        auto *onnx = dynamic_cast<inference::OnnxInference *>(inference_.get());
        if (!onnx) throw std::runtime_error("Unsupported inference type");
        results = onnx->InferBatchSegmentation(images);
    }

    size_t get_batch() const {
        if (!inference_) throw std::runtime_error("Segmenter not initialized");
        return static_cast<size_t>(inference_->GetConfig().batch_size);
    }

    size_t get_input_width() const {
        if (!inference_) throw std::runtime_error("Segmenter not initialized");
        auto cfg = inference_->GetConfig();
        return static_cast<size_t>(cfg.img_scale.width > 0 ? cfg.img_scale.width : cfg.class_scale);
    }

    size_t get_input_height() const {
        if (!inference_) throw std::runtime_error("Segmenter not initialized");
        auto cfg = inference_->GetConfig();
        return static_cast<size_t>(cfg.img_scale.height > 0 ? cfg.img_scale.height : cfg.class_scale);
    }

private:
    std::unique_ptr<inference::BaseInference> inference_;
};

Segmenter::Segmenter(const std::string &model_path, const std::string &label_path,
                     int device_id, const std::string &config_path)
    : m_pHandle(std::make_shared<SegmenterPrivate>(model_path, label_path, device_id, config_path)) {}

Segmenter::~Segmenter() = default;

void Segmenter::process(std::vector<cv::Mat> &images, std::vector<SegmentationResult> &results) {
    m_pHandle->process(images, results);
}

size_t Segmenter::GetBatch() const { return m_pHandle->get_batch(); }
size_t Segmenter::GetInputWidth() const { return m_pHandle->get_input_width(); }
size_t Segmenter::GetInputHeight() const { return m_pHandle->get_input_height(); }

} // namespace JHDeepCore
