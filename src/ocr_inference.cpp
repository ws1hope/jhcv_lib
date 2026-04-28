#include "ocr_inference.h"
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include "src/api/pipelines/ocr.h"
#include "src/pipelines/ocr/result.h"

struct OCRInference::Impl {
    std::unique_ptr<PaddleOCR> ocr_engine;
};

OCRInference::OCRInference(const Params& params) : impl_(std::make_unique<Impl>()) {
    PaddleOCRParams ocr_params;
    ocr_params.text_detection_model_dir = params.text_detection_model_dir;
    ocr_params.text_recognition_model_dir = params.text_recognition_model_dir;
    ocr_params.device = params.device;
    ocr_params.use_doc_orientation_classify = params.use_doc_orientation_classify;
    ocr_params.use_doc_unwarping = params.use_doc_unwarping;
    ocr_params.use_textline_orientation = params.use_textline_orientation;
    ocr_params.text_det_thresh = params.text_det_thresh;
    ocr_params.text_det_box_thresh = params.text_det_box_thresh;
    ocr_params.text_det_unclip_ratio = params.text_det_unclip_ratio;
    ocr_params.text_rec_score_thresh = params.text_rec_score_thresh;
    ocr_params.cpu_threads = params.cpu_threads;
    ocr_params.enable_mkldnn = params.enable_mkldnn;

    impl_->ocr_engine = std::make_unique<PaddleOCR>(ocr_params);
}

OCRInference::~OCRInference() = default;

OCRDetectResult OCRInference::predict(const std::string& image_path) {
    auto results = impl_->ocr_engine->Predict(image_path);
    OCRDetectResult result;

    if (!results.empty()) {
        auto* ocr_res = dynamic_cast<::OCRResult*>(results[0].get());
        if (ocr_res) {
            const auto& pipe_result = ocr_res->GetPipelineResult();
            const auto& texts = pipe_result.rec_texts;
            const auto& scores = pipe_result.rec_scores;
            const auto& polys = pipe_result.rec_polys;

            size_t count = texts.size();
            result.boxes.resize(count);
            for (size_t i = 0; i < count; ++i) {
                result.boxes[i].text = texts[i];
                result.boxes[i].confidence = (i < scores.size()) ? scores[i] : 0.0f;
                if (i < polys.size()) {
                    for (const auto& pt : polys[i]) {
                        result.boxes[i].points.emplace_back(pt.x, pt.y);
                    }
                }
            }
        }
    }

    return result;
}

OCRDetectResult OCRInference::predict(const cv::Mat& image) {
    std::string temp_path = "temp_ocr_input.png";
    cv::imwrite(temp_path, image);
    auto result = predict(temp_path);
    std::remove(temp_path.c_str());
    return result;
}