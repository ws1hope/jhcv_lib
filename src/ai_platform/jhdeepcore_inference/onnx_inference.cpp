#include "jhdeepcore_inference/onnx_inference.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace JHDeepCore {
namespace inference {

namespace {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
bool bench_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("JHDEEP_BENCH");
        return env && std::string(env) == "1";
    }();
    return enabled;
}

bool h2d_split_enabled() {
    static const bool enabled = []() {
        const char *env = std::getenv("JHDEEP_H2D_SPLIT");
        return env && std::string(env) == "1";
    }();
    return enabled;
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

void log_pure_inference(double ms) {
    std::cerr << "[BENCH] Compute phase (session->Run): " << ms << " ms" << std::endl;
}

// JHDEEP_BENCH=1 时按段打印预处理各步耗时；关闭时仅一次 now() 调用，开销可忽略。
// 用 steady_clock（保证单调，不受系统时钟调整影响）测间隔。
struct PreStepTimer {
    const char *tag;
    std::chrono::steady_clock::time_point t0;
    explicit PreStepTimer(const char *t) : tag(t), t0(std::chrono::steady_clock::now()) {}
    ~PreStepTimer() {
        if (!bench_enabled()) return;
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        std::cerr << "[BENCH] " << tag << ": " << ms << " ms" << std::endl;
    }
};

#ifdef USE_CUDA
enum GpuEventIndex : size_t {
    H2D_START = 0,
    H2D_END,
    COMPUTE_START,
    COMPUTE_END,
    D2H_START,
    D2H_END,
    GPU_EVENT_COUNT
};

static void checkCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

static cudaEvent_t asCudaEvent(void *event) {
    return static_cast<cudaEvent_t>(event);
}

static double elapsedCudaMs(void *start, void *end) {
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, asCudaEvent(start), asCudaEvent(end)),
              "cudaEventElapsedTime");
    return static_cast<double>(ms);
}
#endif

static double steadyNowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

static auto to_model_path(const std::string &s) {
#ifdef _WIN32
    std::wstring ws;
    if (!s.empty()) {
        int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
        if (sz > 0) {
            ws.resize(sz - 1);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], sz);
        } else {
            ws = std::wstring(s.begin(), s.end());
        }
    }
    return ws;
#else
    return s;
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
    std::cerr << "[DEBUG] OnnxInference body entered, device=" << device_ << std::endl;
#ifdef ONNXRUNTIME_FOUND
    session_options_.SetIntraOpNumThreads(8);
    session_options_.SetInterOpNumThreads(8);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    session_options_.AddConfigEntry("session.enable_mem_pattern", "0");
    session_options_.AddConfigEntry("session.enable_mem_reuse", "1");

    if (device_ == "cuda") {
#ifdef USE_CUDA
        // split 路径：建一条 CUDA stream 并让 ORT 的 CUDA EP 跑在上面（user_compute_stream），
        // 这样 H2D/compute/D2H 全在 kernel_stream_ 上，可用 event 得到互不重叠的阶段时间。
        if (h2d_split_enabled()) {
            cudaStream_t s = nullptr;
            const cudaError_t status = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            if (status == cudaSuccess) {
                kernel_stream_ = s;
            } else {
                std::cerr << "[WARN] create CUDA timing stream failed; split timing disabled: "
                          << cudaGetErrorString(status) << std::endl;
            }
        }
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
            cuda_options.arena_extend_strategy = 0;
            cuda_options.do_copy_in_default_stream = 1;
            cuda_options.has_user_compute_stream = 0;
            cuda_options.user_compute_stream = nullptr;
            if (kernel_stream_) {
                cuda_options.has_user_compute_stream = 1;
                cuda_options.user_compute_stream = kernel_stream_;
            }
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[INFO] ONNX Runtime: using CUDA provider"
                      << (kernel_stream_ ? " (user_compute_stream for split timing)" : "") << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[WARN] CUDA provider not available, falling back to CPU: " << e.what() << std::endl;
            device_ = "cpu";
        }
#else
        std::cerr << "[WARN] CUDA not compiled in, using optimized CPU for Mac M-series" << std::endl;
        device_ = "cpu";
#endif
    } else if (device_ == "cpu") {
        std::cout << "[INFO] ONNX Runtime: using optimized CPU for Mac M-series chips" << std::endl;
    }
#endif
}

OnnxInference::~OnnxInference() {
#ifdef USE_CUDA
    destroyGpuTimingResources();
#endif
#ifdef ONNXRUNTIME_FOUND
    cuda_allocator_.reset();
    session_.reset();   // 释放 allocator/session 后再销毁 EP 使用的 user stream
#endif
#ifdef USE_CUDA
    if (kernel_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(kernel_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(kernel_stream_));
        kernel_stream_ = nullptr;
    }
#endif
}

bool OnnxInference::LoadModel() {
#ifdef ONNXRUNTIME_FOUND
    try {
        auto model_path = to_model_path(model_path_);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes > 0) {
            auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
            input_name_ = std::string(input_name_allocated.get());

            Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = input_tensor_info.GetShape();

            // 动态尺寸模型（-1 3 -1 -1）从 YAML 配置解析 H/W
            if (input_shape_.size() >= 4) {
                cv::Size target = config_.img_scale.width > 0 ? config_.img_scale : cv::Size(640, 640);
                if (input_shape_[0] == -1) input_shape_[0] = 1;
                if (input_shape_[2] == -1) input_shape_[2] = target.height;
                if (input_shape_[3] == -1) input_shape_[3] = target.width;
            }
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

        // 预分配推理输入缓冲区
        size_t input_size = 1;
        for (int64_t dim : input_shape_) {
            input_size *= static_cast<size_t>(dim);
        }
        input_buffer_.resize(input_size);
        std::cerr << "[DEBUG] LoadModel: input_shape=[";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cerr << input_shape_[i] << (i + 1 < input_shape_.size() ? "," : "");
        }
        std::cerr << "] input_buffer_size=" << input_size << std::endl;

        // 预构建 C 字符串指针（推理时不再重复构建）
        output_names_cstr_.clear();
        for (const auto &name : output_names_) {
            output_names_cstr_.push_back(name.c_str());
        }
        input_names_cstr_ = {input_name_.c_str()};

        model_loaded_ = true;
        std::cerr << "[DEBUG] LoadModel done" << std::endl;

        // split 路径：拿 CUDA EP 的 allocator（供 prepareInput 分配 GPU 输入 buffer）。
        // 用 "Cuda"/OrtArenaAllocator 向 session 查 EP 注册的 allocator；失败则 split 路径回退 CPU。
        if (device_ == "cuda" && h2d_split_enabled()) {
            try {
                cuda_allocator_ = std::make_unique<Ort::Allocator>(
                    *session_, Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault));
                std::cerr << "[INFO] CUDA allocator ready for split timing" << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "[WARN] get CUDA allocator failed, split falls back to CPU: " << e.what() << std::endl;
                cuda_allocator_.reset();
            }
        }

#ifdef USE_CUDA
        gpu_timing_ready_ = kernel_stream_ && cuda_allocator_ && createGpuTimingEvents();
        if (h2d_split_enabled() && !gpu_timing_ready_) {
            std::cerr << "[WARN] GPU split timing is not fully initialized; using unsplit timing"
                      << std::endl;
        }
#endif

        if (warmup_enabled_ && device_ == "cuda") {
            WarmupModel();
            // warmup 只用于稳定 CUDA/ORT，不得污染第一次业务推理的统计。
            batch_timing_ = InferenceTiming{};
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

    auto _pre0 = std::chrono::steady_clock::now();
    cv::Mat preprocessed = PreprocessImageCommon(image);
    PreprocessForOnnx(preprocessed);
    auto _pre1 = std::chrono::steady_clock::now();

    std::vector<float> output = RunInference(input_buffer_);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    // tensor_ms / run_ms 由 RunInference 内部累加

    return ProcessClassificationOutput(output);
}

std::vector<ClassificationResult> OnnxInference::InferBatch(const std::vector<cv::Mat> &images) {
    batch_timing_ = InferenceTiming{};
    batch_timing_.device = device_;
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

    auto _pre0 = std::chrono::steady_clock::now();
    cv::Mat preprocessed = PreprocessImageCommon(image);
    PreprocessForOnnx(preprocessed);
    auto _pre1 = std::chrono::steady_clock::now();
#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }
    float *input_ptr = prepareInput(input_buffer_.data(), input_buffer_.size());

    auto _ten0 = std::chrono::steady_clock::now();
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(inputMemInfoPtr(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    double _run_ms = 0;
    auto output_tensors = executeRun(input_tensor, _run_ms);

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    std::vector<float> output;
    readOutput(float_array, output_size, output, _run_ms);
    if (bench_enabled()) log_pure_inference(_run_ms);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
    if (bench_enabled() || input_on_gpu_) {
        std::cerr << "[BENCH] h2d=" << last_h2d_ms_ << " run=" << _run_ms
                  << " d2h=" << last_d2h_ms_ << " gpu_total=" << last_gpu_total_ms_
                  << " ms" << std::endl;
    }
    freeInput(input_ptr);
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessSegmentationOutput(output, output_shape);
}

std::vector<SegmentationResult> OnnxInference::InferBatchSegmentation(const std::vector<cv::Mat> &images) {
    batch_timing_ = InferenceTiming{};
    batch_timing_.device = device_;
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

    auto _pre0 = std::chrono::steady_clock::now();
    cv::Mat preprocessed = PreprocessImageDetection(image);
    PreprocessForOnnx(preprocessed);
    auto _pre1 = std::chrono::steady_clock::now();
#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }
    float *input_ptr = prepareInput(input_buffer_.data(), input_buffer_.size());

    auto _ten0 = std::chrono::steady_clock::now();
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(inputMemInfoPtr(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    double _run_ms = 0;
    auto output_tensors = executeRun(input_tensor, _run_ms);

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    std::vector<float> output;
    readOutput(float_array, output_size, output, _run_ms);
    if (bench_enabled()) log_pure_inference(_run_ms);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
    if (bench_enabled() || input_on_gpu_) {
        std::cerr << "[BENCH] h2d=" << last_h2d_ms_ << " run=" << _run_ms
                  << " d2h=" << last_d2h_ms_ << " gpu_total=" << last_gpu_total_ms_
                  << " ms" << std::endl;
    }
    freeInput(input_ptr);
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessDetectionOutput(output, output_shape);
}

std::vector<DetectionResult> OnnxInference::InferBatchDetection(const std::vector<cv::Mat> &images) {
    batch_timing_ = InferenceTiming{};
    batch_timing_.device = device_;
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

    auto _pre0 = std::chrono::steady_clock::now();
    cv::Mat preprocessed = PreprocessImageDetection(image);
    PreprocessForOnnx(preprocessed);
    auto _pre1 = std::chrono::steady_clock::now();
#ifdef ONNXRUNTIME_FOUND
    if (!session_) {
        throw std::runtime_error("Model session not initialized");
    }
    float *input_ptr = prepareInput(input_buffer_.data(), input_buffer_.size());

    auto _ten0 = std::chrono::steady_clock::now();
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(inputMemInfoPtr(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    double _run_ms = 0;
    auto output_tensors = executeRun(input_tensor, _run_ms);

    if (output_tensors.empty() || output_tensors.size() < 2) {
        throw std::runtime_error("Instance segmentation model should have two outputs");
    }

    float *detection_array = output_tensors[0].GetTensorMutableData<float>();
    auto detection_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t detection_size = detection_tensor_info.GetElementCount();
    std::vector<int64_t> detection_output_shape = detection_tensor_info.GetShape();
    std::vector<float> detection_output;

    float *protos_array = output_tensors[1].GetTensorMutableData<float>();
    auto protos_tensor_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    size_t protos_size = protos_tensor_info.GetElementCount();
    std::vector<int64_t> protos_output_shape = protos_tensor_info.GetShape();
    std::vector<float> protos_output;
    readOutputs({detection_array, protos_array}, {detection_size, protos_size},
                {&detection_output, &protos_output}, _run_ms);
    if (bench_enabled()) log_pure_inference(_run_ms);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
    if (bench_enabled() || input_on_gpu_) {
        std::cerr << "[BENCH] h2d=" << last_h2d_ms_ << " run=" << _run_ms
                  << " d2h=" << last_d2h_ms_ << " gpu_total=" << last_gpu_total_ms_
                  << " ms" << std::endl;
    }
    freeInput(input_ptr);
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif

    return ProcessInstanceSegmentationOutput(detection_output, detection_output_shape, protos_output,
                                             protos_output_shape);
}

std::vector<InstanceSegmentationResult> OnnxInference::InferBatchInstanceSegmentation(
    const std::vector<cv::Mat> &images) {
    batch_timing_ = InferenceTiming{};
    batch_timing_.device = device_;
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
        std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
        for (int i = 0; i < iterations; ++i) {
            RunInference(input_buffer_);
        }
    } catch (const std::exception &e) {
        std::cerr << "Warning: GPU warmup failed: " << e.what() << std::endl;
    }
#endif
}

void OnnxInference::PreprocessForOnnx(const cv::Mat &image) {
    const cv::Mat *img = &image;
    cv::Mat resized;
    if (input_shape_.size() >= 4) {
        int expected_h = static_cast<int>(input_shape_[2]);
        int expected_w = static_cast<int>(input_shape_[3]);
        if (image.rows != expected_h || image.cols != expected_w) {
            PreStepTimer _("ForOnnx.resize");
            cv::resize(image, resized, cv::Size(expected_w, expected_h));
            img = &resized;
        }
    }

    int channels = img->channels();
    int height = img->rows;
    int width = img->cols;
    int hw = height * width;

    {
        PreStepTimer _("ForOnnx.transpose");
        for (int h = 0; h < height; ++h) {
            const cv::Vec3f *row = img->ptr<cv::Vec3f>(h);
            for (int w = 0; w < width; ++w) {
                const cv::Vec3f &pixel = row[w];
                int spatial = h * width + w;
                input_buffer_[spatial] = pixel[0];
                input_buffer_[hw + spatial] = pixel[1];
                input_buffer_[2 * hw + spatial] = pixel[2];
            }
        }
    }
}

std::vector<float> OnnxInference::RunInference(const std::vector<float> &input_data) {
#ifdef ONNXRUNTIME_FOUND
    if (!model_loaded_ || !session_) {
        throw std::runtime_error("Model not loaded");
    }

    float *input_ptr = prepareInput(input_data.data(), input_data.size());

    auto _ten0 = std::chrono::steady_clock::now();
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(inputMemInfoPtr(), input_ptr, input_data.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    double _run_ms = 0;
    auto output_tensors = executeRun(input_tensor, _run_ms);

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    std::vector<float> output;
    readOutput(float_array, output_size, output, _run_ms);
    if (bench_enabled()) log_pure_inference(_run_ms);

    // 分类路径：tensor/run 在此累加；count/preprocess 由 InferSingle 累加
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
    if (bench_enabled() || input_on_gpu_) {
        std::cerr << "[BENCH] h2d=" << last_h2d_ms_ << " run=" << _run_ms
                  << " d2h=" << last_d2h_ms_ << " gpu_total=" << last_gpu_total_ms_
                  << " ms" << std::endl;
    }
    freeInput(input_ptr);
    return output;
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif
}

bool OnnxInference::useGpuTensor() const {
#if defined(USE_CUDA) && defined(ONNXRUNTIME_FOUND)
    return device_ == "cuda" && h2d_split_enabled() && kernel_stream_ && cuda_allocator_ &&
           gpu_timing_ready_;
#else
    return false;
#endif
}

#ifdef ONNXRUNTIME_FOUND
const OrtMemoryInfo *OnnxInference::inputMemInfoPtr() const {
    if (input_on_gpu_ && cuda_allocator_) return cuda_allocator_->GetInfo();
    return memory_info_;
}

std::vector<Ort::Value> OnnxInference::executeRun(Ort::Value &input_tensor, double &run_ms) {
    if (gpu_timing_active_ && input_on_gpu_ && cuda_allocator_) {
        // split：IoBinding 把输出 bind 到 GPU（cuda_allocator_ 的 MemoryInfo），ORT 不再 D2H 输出，
        // compute event 只包围 Run；binding 的 host 开销不混入 compute phase。
        Ort::IoBinding binding(*session_);
        binding.BindInput(input_names_cstr_[0], input_tensor);
        for (size_t i = 0; i < output_names_cstr_.size(); ++i) {
            binding.BindOutput(output_names_cstr_[i], cuda_allocator_->GetInfo());
        }
#ifdef USE_CUDA
        cudaStream_t stream = static_cast<cudaStream_t>(kernel_stream_);
        checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[COMPUTE_START]), stream),
                  "record compute start");
#endif
        session_->Run(Ort::RunOptions{nullptr}, binding);
#ifdef USE_CUDA
        checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[COMPUTE_END]), stream),
                  "record compute end");
#endif
        // CUDA event 统一在 D2H 完成后读取，避免在三个阶段之间各做一次同步。
        run_ms = 0.0;
        return binding.GetOutputValues();
    }

    const auto start = std::chrono::steady_clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                                 output_names_cstr_.data(), output_names_cstr_.size());
    run_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start)
                 .count();
    return outputs;
}
#endif

float *OnnxInference::prepareInput(const float *data, size_t count) {
    input_on_gpu_ = false;
    gpu_timing_active_ = false;
    last_h2d_ms_ = 0;
    last_d2h_ms_ = 0;
    last_gpu_total_ms_ = 0;
    inference_wall_start_ms_ = steadyNowMs();

    // 非 split：原样返回 CPU 指针供建 CPU tensor（ORT 内部自行 H2D）。
    if (!useGpuTensor()) return const_cast<float *>(data);
#ifdef USE_CUDA
    if (count == 0 || !ensurePinnedInput(count) || !ensureCudaInput(count)) {
        return const_cast<float *>(data);
    }

    std::memcpy(pinned_input_, data, count * sizeof(float));
    cudaStream_t stream = static_cast<cudaStream_t>(kernel_stream_);
    checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[H2D_START]), stream),
              "record H2D start");
    checkCuda(cudaMemcpyAsync(cuda_input_, pinned_input_, count * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync H2D");
    checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[H2D_END]), stream),
              "record H2D end");

    input_on_gpu_ = true;
    gpu_timing_active_ = true;
    return static_cast<float *>(cuda_input_);
#else
    return const_cast<float *>(data);
#endif
}

void OnnxInference::freeInput(float *ptr) {
    (void)ptr;
    // cuda_input_ 是复用成员缓冲，仅在析构或扩容时释放。
    input_on_gpu_ = false;
    gpu_timing_active_ = false;
}

void OnnxInference::readOutput(const float *src, size_t count, std::vector<float> &dst,
                               double &run_ms) {
    readOutputs({src}, {count}, {&dst}, run_ms);
}

void OnnxInference::readOutputs(const std::vector<const float *> &srcs,
                                const std::vector<size_t> &counts,
                                const std::vector<std::vector<float> *> &dsts,
                                double &run_ms) {
    if (srcs.size() != counts.size() || srcs.size() != dsts.size()) {
        throw std::invalid_argument("readOutputs arguments have different sizes");
    }

    size_t total_count = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (!dsts[i]) throw std::invalid_argument("readOutputs destination is null");
        if (counts[i] > std::numeric_limits<size_t>::max() - total_count) {
            throw std::overflow_error("readOutputs element count overflow");
        }
        total_count += counts[i];
        dsts[i]->resize(counts[i]);
    }

    if (!input_on_gpu_) {
        // src 为 CPU 指针（非 split：ORT 内部已 D2H 到 CPU），直接拷贝
        for (size_t i = 0; i < srcs.size(); ++i) {
            if (counts[i] > 0) {
                std::copy(srcs[i], srcs[i] + counts[i], dsts[i]->begin());
            }
        }
        batch_timing_.wall_ms += steadyNowMs() - inference_wall_start_ms_;
        return;
    }
#ifdef USE_CUDA
    if (!gpu_timing_active_) {
        throw std::runtime_error("GPU output received without an active timing context");
    }
    if (!ensurePinnedOutput(total_count)) {
        throw std::runtime_error("failed to allocate pinned output buffer");
    }

    cudaStream_t stream = static_cast<cudaStream_t>(kernel_stream_);
    checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[D2H_START]), stream),
              "record D2H start");
    size_t offset = 0;
    float *pinned = static_cast<float *>(pinned_output_);
    for (size_t i = 0; i < srcs.size(); ++i) {
        if (counts[i] > 0) {
            checkCuda(cudaMemcpyAsync(pinned + offset, srcs[i], counts[i] * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync D2H");
        }
        offset += counts[i];
    }
    checkCuda(cudaEventRecord(asCudaEvent(gpu_events_[D2H_END]), stream),
              "record D2H end");
    checkCuda(cudaEventSynchronize(asCudaEvent(gpu_events_[D2H_END])),
              "synchronize D2H end");

    last_h2d_ms_ = elapsedCudaMs(gpu_events_[H2D_START], gpu_events_[H2D_END]);
    run_ms = elapsedCudaMs(gpu_events_[COMPUTE_START], gpu_events_[COMPUTE_END]);
    last_d2h_ms_ = elapsedCudaMs(gpu_events_[D2H_START], gpu_events_[D2H_END]);
    last_gpu_total_ms_ = elapsedCudaMs(gpu_events_[H2D_START], gpu_events_[D2H_END]);

    offset = 0;
    for (size_t i = 0; i < dsts.size(); ++i) {
        if (counts[i] > 0) {
            std::copy(pinned + offset, pinned + offset + counts[i], dsts[i]->begin());
        }
        offset += counts[i];
    }

    batch_timing_.h2d_ms += last_h2d_ms_;
    batch_timing_.d2h_ms += last_d2h_ms_;
    batch_timing_.gpu_total_ms += last_gpu_total_ms_;
    batch_timing_.wall_ms += steadyNowMs() - inference_wall_start_ms_;
    batch_timing_.h2d_split = true;
    batch_timing_.gpu_timing_valid = true;
    gpu_timing_active_ = false;
#else
    (void)total_count;
#endif
}

bool OnnxInference::createGpuTimingEvents() {
#ifdef USE_CUDA
    for (size_t i = 0; i < GPU_EVENT_COUNT; ++i) {
        cudaEvent_t event = nullptr;
        const cudaError_t status = cudaEventCreate(&event);
        if (status != cudaSuccess) {
            std::cerr << "[WARN] create CUDA timing event failed: "
                      << cudaGetErrorString(status) << std::endl;
            for (size_t j = 0; j < i; ++j) {
                cudaEventDestroy(asCudaEvent(gpu_events_[j]));
                gpu_events_[j] = nullptr;
            }
            return false;
        }
        gpu_events_[i] = event;
    }
    return true;
#else
    return false;
#endif
}

bool OnnxInference::ensurePinnedInput(size_t count) {
#ifdef USE_CUDA
    if (count <= pinned_input_count_ && pinned_input_) return true;
    void *next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, count * sizeof(float), cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "[WARN] allocate pinned input failed; split timing disabled: "
                  << cudaGetErrorString(status) << std::endl;
        gpu_timing_ready_ = false;
        return false;
    }
    if (pinned_input_) cudaFreeHost(pinned_input_);
    pinned_input_ = next;
    pinned_input_count_ = count;
    return true;
#else
    (void)count;
    return false;
#endif
}

bool OnnxInference::ensurePinnedOutput(size_t count) {
#ifdef USE_CUDA
    if (count == 0) return true;
    if (count <= pinned_output_count_ && pinned_output_) return true;
    void *next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, count * sizeof(float), cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "[ERROR] allocate pinned output failed: " << cudaGetErrorString(status)
                  << std::endl;
        return false;
    }
    if (pinned_output_) cudaFreeHost(pinned_output_);
    pinned_output_ = next;
    pinned_output_count_ = count;
    return true;
#else
    (void)count;
    return false;
#endif
}

bool OnnxInference::ensureCudaInput(size_t count) {
#if defined(USE_CUDA) && defined(ONNXRUNTIME_FOUND)
    if (count <= cuda_input_count_ && cuda_input_) return true;
    void *next = nullptr;
    try {
        next = cuda_allocator_ ? cuda_allocator_->Alloc(count * sizeof(float)) : nullptr;
    } catch (const std::exception &e) {
        std::cerr << "[WARN] allocate CUDA input failed; split timing disabled: " << e.what()
                  << std::endl;
    }
    if (!next) {
        gpu_timing_ready_ = false;
        return false;
    }
    if (cuda_input_ && cuda_allocator_) cuda_allocator_->Free(cuda_input_);
    cuda_input_ = next;
    cuda_input_count_ = count;
    return true;
#else
    (void)count;
    return false;
#endif
}

void OnnxInference::destroyGpuTimingResources() {
#ifdef USE_CUDA
    gpu_timing_active_ = false;
    gpu_timing_ready_ = false;
    if (kernel_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(kernel_stream_));
    }
#ifdef ONNXRUNTIME_FOUND
    if (cuda_input_ && cuda_allocator_) cuda_allocator_->Free(cuda_input_);
#endif
    cuda_input_ = nullptr;
    cuda_input_count_ = 0;
    if (pinned_input_) cudaFreeHost(pinned_input_);
    pinned_input_ = nullptr;
    pinned_input_count_ = 0;
    if (pinned_output_) cudaFreeHost(pinned_output_);
    pinned_output_ = nullptr;
    pinned_output_count_ = 0;
    for (void *&event : gpu_events_) {
        if (event) cudaEventDestroy(asCudaEvent(event));
        event = nullptr;
    }
#endif
}

} // namespace inference
} // namespace JHDeepCore
