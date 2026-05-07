#include "jhdeepcore_instance_segment/instance_segmenter.h"
#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <stdexcept>

namespace JHDeepCore {
namespace instance_segment {

class InstanceSegmenterImpl::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device,
         const std::vector<std::string> &class_names)
        : inference_(inference::InferenceFactory::CreateInference(model_path, device, class_names)) {}

    std::unique_ptr<inference::BaseInference> inference_;
};

InstanceSegmenterImpl::InstanceSegmenterImpl(const std::string &model_path, const std::string &device,
                                             const std::vector<std::string> &class_names)
    : pImpl_(std::make_unique<Impl>(model_path, device, class_names)) {}

InstanceSegmenterImpl::~InstanceSegmenterImpl() = default;

JHDeepCore::InstanceSegmentationResult InstanceSegmenterImpl::SegmentSingle(const cv::Mat &image) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Instance segmenter not initialized");
    }

    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferSingleInstanceSegmentation(image);
    }

    throw std::runtime_error("Unsupported inference type");
}

std::vector<JHDeepCore::InstanceSegmentationResult> InstanceSegmenterImpl::SegmentBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_ || !pImpl_->inference_) {
        throw std::runtime_error("Instance segmenter not initialized");
    }

    if (images.empty()) {
        return {};
    }

    auto *onnx_inference = dynamic_cast<inference::OnnxInference *>(pImpl_->inference_.get());
    if (onnx_inference) {
        return onnx_inference->InferBatchInstanceSegmentation(images);
    }

    throw std::runtime_error("Unsupported inference type");
}

} // namespace instance_segment
} // namespace JHDeepCore
