#include "jhdeepcore_classify/classifier.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {
namespace classify {

class ClassifierImpl::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device,
         const std::vector<std::string> &class_names)
        : inference_(inference::InferenceFactory::CreateInference(model_path, device, class_names)) {}

    std::unique_ptr<inference::BaseInference> inference_;
};

ClassifierImpl::ClassifierImpl(const std::string &model_path, const std::string &device,
                               const std::vector<std::string> &class_names)
    : pImpl_(std::make_unique<Impl>(model_path, device, class_names)) {}

ClassifierImpl::~ClassifierImpl() = default;

JHDeepCore::ClassificationResult ClassifierImpl::ClassifySingle(const cv::Mat &image) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Classifier not initialized");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferSingle(image);
    }

    throw std::runtime_error("Unsupported inference type");
}

std::vector<JHDeepCore::ClassificationResult> ClassifierImpl::ClassifyBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Classifier not initialized");
    }

    if (images.empty()) {
        return {};
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferBatch(images);
    }

    throw std::runtime_error("Unsupported inference type");
}

} // namespace classify
} // namespace JHDeepCore
