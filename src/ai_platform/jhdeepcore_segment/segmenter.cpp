#include "jhdeepcore_segment/segmenter.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {
namespace segment {

class SegmenterImpl::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device,
         const std::vector<std::string> &class_names)
        : inference_(inference::InferenceFactory::CreateInference(model_path, device, class_names)) {}

    std::unique_ptr<inference::BaseInference> inference_;
};

SegmenterImpl::SegmenterImpl(const std::string &model_path, const std::string &device,
                             const std::vector<std::string> &class_names)
    : pImpl_(std::make_unique<Impl>(model_path, device, class_names)) {}

SegmenterImpl::~SegmenterImpl() = default;

JHDeepCore::SegmentationResult SegmenterImpl::SegmentSingle(const cv::Mat &image) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Segmenter not initialized");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferSingleSegmentation(image);
    }

    throw std::runtime_error("Unsupported inference type");
}

std::vector<JHDeepCore::SegmentationResult> SegmenterImpl::SegmentBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Segmenter not initialized");
    }

    if (images.empty()) {
        return {};
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferBatchSegmentation(images);
    }

    throw std::runtime_error("Unsupported inference type");
}

} // namespace segment
} // namespace JHDeepCore
