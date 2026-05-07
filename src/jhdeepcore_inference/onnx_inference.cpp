#include "jhdeepcore_inference/onnx_inference.h"
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace JHDeepCore {
namespace inference {

static std::wstring to_wide(const std::string &s) {
#ifdef _WIN32
    if (s.empty()) return std::wstring();
    int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (sz <= 0) return std::wstring(s.begin(), s.end());
    std::wstring ws(sz - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], sz);
    return ws;
#else
    return std::wstring(s.begin(), s.end());
#endif
}

OnnxInference::OnnxInference(const std::string &model_path, const std::string &device,
                             const std::vector<std::string> &class_names, bool warmup)
    : BaseInference(model_path, device, class_names)
#ifdef ONNXRUNTIME_FOUND
      ,
      env_(ORT_LOGGING_LEVEL_WARNING, "JHDeepCore"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
      ,
      model_loaded_(false), warmup_enabled_(warmup) {
#ifdef ONNXRUNTIME_FOUND
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#endif
}

OnnxInference::~OnnxInference() {
#ifdef ONNXRUNTIME_FOUND
    session_.reset();
#endif
}

bool OnnxInference::LoadModel() {
#ifdef ONNXRUNTIME_FOUND
    try {
        std::wstring wpath = to_wide(model_path_);
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes > 0) {
            auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
            input_name_ = std::string(input_name_allocated.get());

            Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = input_tensor_info.GetShape();
        }

        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.clear();
        output_shapes_.clear();
        for (size_t i = 0; i < num_output_nodes; ++i) {
            auto output_name_allocated = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name_allocated.get()));

            Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(output_tensor_info.GetShape());
        }

        model_loaded_ = true;

        if (warmup_enabled_ && device_ == "cuda") {
            WarmupModel();
        }

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error: Failed to load ONNX model: " << e.what() << std::endl;
        model_loaded_ = false;
        return false;
    }
#else
    std::cerr << "Error: ONNX Runtime not found" << std::endl;
    return false;
#endif
}

ClassificationResult OnnxInference::InferSingle(const cv::Mat &image) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded, call LoadModel() first");
    }

    cv::Mat preprocessed = PreprocessImageCommon(image);
    std::vector<float> input_data = PreprocessForOnnx(preprocessed);

    std::vector<float> output = RunInference(input_data);

    return ProcessClassificationOutput(output);
}

std::vector<ClassificationResult> OnnxInference::InferBatch(const std::vector<cv::Mat> &images) {
    std::vector<ClassificationResult> results;
    for (const auto &image : images) {
        results.push_back(InferSingle(image));
    }
    return results;
}

SegmentationResult OnnxInference::InferSingleSegmentation(const cv::Mat &image) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded, call LoadModel() first");
    }

    cv::Mat preprocessed = PreprocessImageCommon(image);
    std::vector<float> input_data = PreprocessForOnnx(preprocessed);

#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }

    std::vector<int64_t> input_shape = input_shape_;
    if (input_shape[0] == -1) {
        input_shape[0] = 1;
    }

    size_t input_tensor_size = 1;
    for (int64_t dim : input_shape) {
        input_tensor_size *= static_cast<size_t>(dim);
    }

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(input_data.data()), input_data.size(),
                                        input_shape.data(), input_shape.size());

    std::vector<const char *> output_names_cstr;
    for (const auto &name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    std::vector<const char *> input_names_cstr = {input_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    std::vector<int64_t> output_shape = tensor_info.GetShape();

    std::vector<float> output(float_array, float_array + output_size);
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessSegmentationOutput(output, output_shape);
}

std::vector<SegmentationResult> OnnxInference::InferBatchSegmentation(const std::vector<cv::Mat> &images) {
    std::vector<SegmentationResult> results;
    for (const auto &image : images) {
        results.push_back(InferSingleSegmentation(image));
    }
    return results;
}

DetectionResult OnnxInference::InferSingleDetection(const cv::Mat &image) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded, call LoadModel() first");
    }

    cv::Mat preprocessed = PreprocessImageDetection(image);
    std::vector<float> input_data = PreprocessForOnnx(preprocessed);

#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }

    std::vector<int64_t> input_shape = input_shape_;
    if (input_shape[0] == -1) {
        input_shape[0] = 1;
    }

    size_t input_tensor_size = 1;
    for (int64_t dim : input_shape) {
        input_tensor_size *= static_cast<size_t>(dim);
    }

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(input_data.data()), input_data.size(),
                                        input_shape.data(), input_shape.size());

    std::vector<const char *> output_names_cstr;
    for (const auto &name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    std::vector<const char *> input_names_cstr = {input_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    std::vector<int64_t> output_shape = tensor_info.GetShape();

    std::vector<float> output(float_array, float_array + output_size);
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessDetectionOutput(output, output_shape);
}

std::vector<DetectionResult> OnnxInference::InferBatchDetection(const std::vector<cv::Mat> &images) {
    std::vector<DetectionResult> results;
    for (const auto &image : images) {
        results.push_back(InferSingleDetection(image));
    }
    return results;
}

InstanceSegmentationResult OnnxInference::InferSingleInstanceSegmentation(const cv::Mat &image) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded, call LoadModel() first");
    }

    cv::Mat preprocessed = PreprocessImageDetection(image);
    std::vector<float> input_data = PreprocessForOnnx(preprocessed);

#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }

    std::vector<int64_t> input_shape = input_shape_;
    if (input_shape[0] == -1) {
        input_shape[0] = 1;
    }

    size_t input_tensor_size = 1;
    for (int64_t dim : input_shape) {
        input_tensor_size *= static_cast<size_t>(dim);
    }

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(input_data.data()), input_data.size(),
                                        input_shape.data(), input_shape.size());

    std::vector<const char *> output_names_cstr;
    for (const auto &name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    std::vector<const char *> input_names_cstr = {input_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());

    if (output_tensors.empty() || output_tensors.size() < 2) {
        throw std::runtime_error("Instance segmentation model should have two outputs");
    }

    float *detection_array = output_tensors[0].GetTensorMutableData<float>();
    auto detection_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t detection_size = detection_tensor_info.GetElementCount();
    std::vector<int64_t> detection_output_shape = detection_tensor_info.GetShape();
    std::vector<float> detection_output(detection_array, detection_array + detection_size);

    float *protos_array = output_tensors[1].GetTensorMutableData<float>();
    auto protos_tensor_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    size_t protos_size = protos_tensor_info.GetElementCount();
    std::vector<int64_t> protos_output_shape = protos_tensor_info.GetShape();
    std::vector<float> protos_output(protos_array, protos_array + protos_size);

#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessInstanceSegmentationOutput(detection_output, detection_output_shape, protos_output,
                                             protos_output_shape);
}

std::vector<InstanceSegmentationResult> OnnxInference::InferBatchInstanceSegmentation(const std::vector<cv::Mat> &images) {
    std::vector<InstanceSegmentationResult> results;
    for (const auto &image : images) {
        results.push_back(InferSingleInstanceSegmentation(image));
    }
    return results;
}

void OnnxInference::WarmupModel(int iterations) {
#ifdef ONNXRUNTIME_FOUND
    if (!model_loaded_) {
        return;
    }

    try {
        int input_size = 1;
        for (int64_t dim : input_shape_) {
            if (dim > 0) {
                input_size *= static_cast<int>(dim);
            }
        }

        std::vector<float> dummy_input(input_size, 0.0f);

        for (int i = 0; i < iterations; ++i) {
            RunInference(dummy_input);
        }
    } catch (const std::exception &e) {
        std::cerr << "Warning: GPU warmup failed: " << e.what() << std::endl;
    }
#endif
}

std::vector<float> OnnxInference::PreprocessForOnnx(const cv::Mat &image) {
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;

    std::vector<float> input_data(channels * height * width);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                cv::Vec3f pixel = image.at<cv::Vec3f>(h, w);
                int idx = c * height * width + h * width + w;
                input_data[idx] = pixel[c];
            }
        }
    }

    return input_data;
}

std::vector<float> OnnxInference::RunInference(const std::vector<float> &input_data) {
#ifdef ONNXRUNTIME_FOUND
    if (!model_loaded_ || !session_) {
        throw std::runtime_error("Model not loaded");
    }

    std::vector<int64_t> input_shape = input_shape_;
    if (input_shape[0] == -1) {
        input_shape[0] = 1;
    }

    size_t input_tensor_size = 1;
    for (int64_t dim : input_shape) {
        input_tensor_size *= static_cast<size_t>(dim);
    }

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(input_data.data()), input_data.size(),
                                        input_shape.data(), input_shape.size());

    std::vector<const char *> output_names_cstr;
    for (const auto &name : output_names_) {
        output_names_cstr.push_back(name.c_str());
    }

    std::vector<const char *> input_names_cstr = {input_name_.c_str()};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), &input_tensor, 1,
                                        output_names_cstr.data(), output_names_cstr.size());

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    std::vector<float> output(float_array, float_array + output_size);
    return output;
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif
}

} // namespace inference
} // namespace JHDeepCore
