#include "JHDeepCore.h"
#include "jhdeepcore_classify/classifier.h"
#include "jhdeepcore_segment/segmenter.h"
#include "jhdeepcore_detect/detector.h"
#include "jhdeepcore_instance_segment/instance_segmenter.h"
#include "jhdeepcore_ocr/ocr_recognizer.h"
#include "jhdeepcore_utils/device_utils.h"
#include <stdexcept>

namespace JHDeepCore {

// ========== Classifier PIMPL ==========
class Classifier::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device) : classifier_impl_(model_path, device) {}

    classify::ClassifierImpl classifier_impl_;
};

Classifier::Classifier(const std::string &model_path, const std::string &device)
    : pImpl_(std::make_unique<Impl>(model_path, device)) {}

Classifier::~Classifier() = default;

ClassificationResult Classifier::ClassifySingle(const cv::Mat &image) {
    if (!pImpl_) {
        throw std::runtime_error("Classifier not initialized");
    }
    return pImpl_->classifier_impl_.ClassifySingle(image);
}

std::vector<ClassificationResult> Classifier::ClassifyBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_) {
        throw std::runtime_error("Classifier not initialized");
    }
    return pImpl_->classifier_impl_.ClassifyBatch(images);
}

// ========== Segmenter PIMPL ==========
class Segmenter::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device) : segmenter_impl_(model_path, device) {}

    segment::SegmenterImpl segmenter_impl_;
};

Segmenter::Segmenter(const std::string &model_path, const std::string &device)
    : pImpl_(std::make_unique<Impl>(model_path, device)) {}

Segmenter::~Segmenter() = default;

SegmentationResult Segmenter::SegmentSingle(const cv::Mat &image) {
    if (!pImpl_) {
        throw std::runtime_error("Segmenter not initialized");
    }
    return pImpl_->segmenter_impl_.SegmentSingle(image);
}

std::vector<SegmentationResult> Segmenter::SegmentBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_) {
        throw std::runtime_error("Segmenter not initialized");
    }
    return pImpl_->segmenter_impl_.SegmentBatch(images);
}

// ========== Detector PIMPL ==========
class Detector::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device) : detector_impl_(model_path, device) {}

    detect::DetectorImpl detector_impl_;
};

Detector::Detector(const std::string &model_path, const std::string &device)
    : pImpl_(std::make_unique<Impl>(model_path, device)) {}

Detector::~Detector() = default;

DetectionResult Detector::DetectSingle(const cv::Mat &image) {
    if (!pImpl_) {
        throw std::runtime_error("Detector not initialized");
    }
    return pImpl_->detector_impl_.DetectSingle(image);
}

std::vector<DetectionResult> Detector::DetectBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_) {
        throw std::runtime_error("Detector not initialized");
    }
    return pImpl_->detector_impl_.DetectBatch(images);
}

// ========== InstanceSegmenter PIMPL ==========
class InstanceSegmenter::Impl {
  public:
    Impl(const std::string &model_path, const std::string &device, const std::vector<std::string> &class_names)
        : instance_segmenter_impl_(model_path, device, class_names) {}

    instance_segment::InstanceSegmenterImpl instance_segmenter_impl_;
};

InstanceSegmenter::InstanceSegmenter(const std::string &model_path, const std::string &device,
                                     const std::vector<std::string> &class_names)
    : pImpl_(std::make_unique<Impl>(model_path, device, class_names)) {}

InstanceSegmenter::~InstanceSegmenter() = default;

InstanceSegmentationResult InstanceSegmenter::SegmentSingle(const cv::Mat &image) {
    if (!pImpl_) {
        throw std::runtime_error("Instance segmenter not initialized");
    }
    return pImpl_->instance_segmenter_impl_.SegmentSingle(image);
}

std::vector<InstanceSegmentationResult> InstanceSegmenter::SegmentBatch(const std::vector<cv::Mat> &images) {
    if (!pImpl_) {
        throw std::runtime_error("Instance segmenter not initialized");
    }
    return pImpl_->instance_segmenter_impl_.SegmentBatch(images);
}

// ========== OCRRecognizer PIMPL ==========
class OCRRecognizer::Impl {
  public:
    explicit Impl(const OCRRecognizer::Params &params) {
        ocr::OCRRecognizerImpl::Params ocr_params;
        ocr_params.rec_model_path = params.rec_model_path;
        ocr_params.rec_label_path = params.rec_label_path;
        ocr_params.device = params.device;
        ocr_params.rec_score_thresh = params.text_rec_score_thresh;
        ocr_params.useGPU = params.useGPU;
        ocr_params.gpuId = params.gpuId;
        ocr_impl_ = std::make_unique<ocr::OCRRecognizerImpl>(ocr_params);
    }

    std::unique_ptr<ocr::OCRRecognizerImpl> ocr_impl_;
};

OCRRecognizer::OCRRecognizer(const Params &params)
    : pImpl_(std::make_unique<Impl>(params)) {}

OCRRecognizer::~OCRRecognizer() = default;

OCRResult OCRRecognizer::Recognize(const cv::Mat &text_image) {
    if (!pImpl_ || !pImpl_->ocr_impl_) {
        throw std::runtime_error("OCR recognizer not initialized");
    }
    return pImpl_->ocr_impl_->Recognize(text_image);
}

OCRResult OCRRecognizer::Recognize(const std::string &image_path) {
    if (!pImpl_ || !pImpl_->ocr_impl_) {
        throw std::runtime_error("OCR recognizer not initialized");
    }
    return pImpl_->ocr_impl_->Recognize(image_path);
}

// ========== Utility ==========
std::string GetOptimalDevice() {
    return utils::DeviceUtils::GetOptimalDevice();
}

} // namespace JHDeepCore
