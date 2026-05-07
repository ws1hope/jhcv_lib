#include "jhdeepcore_detect/detector.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {
namespace detect {

class DetectorImpl::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device,
         const std::vector<std::string> &class_names)
        : inference_(inference::InferenceFactory::CreateInference(model_path, device, class_names)) {}

    std::unique_ptr<inference::BaseInference> inference_;
};

DetectorImpl::DetectorImpl(const std::string &model_path, const std::string &device,
                           const std::vector<std::string> &class_names)
    : pImpl_(std::make_unique<Impl>(model_path, device, class_names)) {}

DetectorImpl::~DetectorImpl() = default;

JHDeepCore::DetectionResult DetectorImpl::DetectSingle(const cv::Mat &image) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Detector not initialized");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferSingleDetection(image);
    }

    throw std::runtime_error("Unsupported inference type");
}

std::vector<JHDeepCore::DetectionResult> DetectorImpl::DetectBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Detector not initialized");
    }

    if (images.empty()) {
        return {};
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferBatchDetection(images);
    }

    throw std::runtime_error("Unsupported inference type");
}

} // namespace detect
} // namespace JHDeepCore
