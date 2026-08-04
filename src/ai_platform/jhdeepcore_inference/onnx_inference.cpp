#include "jhdeepcore_inference/onnx_inference.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
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
    std::cerr << "[BENCH] Pure inference (session->Run): " << ms << " ms" << std::endl;
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

// session->Run() 计时守卫：cuda 设备用 cudaEvent（在默认 stream 0 上 record start/stop，
// 包住同步的 Run()，cudaEventSynchronize 后取 GPU 时间戳 elapsed），cpu 用 steady_clock。
// 注意：ORT 的 CUDA EP kernel 跑在自有 stream，事件记在 stream 0（空闲）故测的是 Run 的 host
// 墙钟，非纯 kernel GPU 时间。split 路径输入在 GPU，run_ms 已排除 H2D/D2H，仍含 ORT host
// 开销+kernel；非 split 路径输入为 CPU tensor，run_ms 含 ORT 内部 H2D+kernel+D2H。事件按调用
// create/destroy，开销 μs 级，相对 ms 级推理可忽略，且只在 device_=="cuda" 时发生。
struct RunTimer {
    bool use_events;
    std::chrono::steady_clock::time_point t0;
#ifdef USE_CUDA
    cudaEvent_t start_ev, stop_ev;
#endif
    explicit RunTimer(bool use_cuda_events) : use_events(use_cuda_events) {
#ifdef USE_CUDA
        if (use_events) {
            cudaEventCreate(&start_ev);
            cudaEventCreate(&stop_ev);
            cudaEventRecord(start_ev, 0);
            return;
        }
#endif
        t0 = std::chrono::steady_clock::now();
    }
    double elapsed_ms() {
#ifdef USE_CUDA
        if (use_events) {
            cudaEventRecord(stop_ev, 0);
            cudaEventSynchronize(stop_ev);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_ev, stop_ev);
            cudaEventDestroy(start_ev);
            cudaEventDestroy(stop_ev);
            return static_cast<double>(ms);
        }
#endif
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

// cudaEvent 计时一段 cudaMemcpyAsync（stream 0 上 record start/stop + sync 后取 elapsed），
// 用于真实 H2D/D2H 计时。事件按调用 create/destroy，开销 μs 级。仅 USE_CUDA 下可用。
#ifdef USE_CUDA
static double timedCudaCopyAsync(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    cudaMemcpyAsync(dst, src, bytes, kind, 0);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return static_cast<double>(ms);
}
#endif
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
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      cuda_mem_info_(Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault))
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
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[INFO] ONNX Runtime: using CUDA provider" << std::endl;
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
    if (cuda_input_) cudaFree(cuda_input_);
#endif
#ifdef ONNXRUNTIME_FOUND
    session_.reset();
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
        Ort::Value::CreateTensor<float>(inputMemInfo(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    RunTimer _run_rt(device_ == "cuda");
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                                        output_names_cstr_.data(), output_names_cstr_.size());
    double _run_ms = _run_rt.elapsed_ms();
    if (bench_enabled()) {
        log_pure_inference(_run_ms);
    }

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    std::vector<float> output;
    readOutput(float_array, output_size, output);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
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
        Ort::Value::CreateTensor<float>(inputMemInfo(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    RunTimer _run_rt(device_ == "cuda");
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                                        output_names_cstr_.data(), output_names_cstr_.size());
    double _run_ms = _run_rt.elapsed_ms();
    if (bench_enabled()) {
        log_pure_inference(_run_ms);
    }

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    std::vector<float> output;
    readOutput(float_array, output_size, output);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
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
        Ort::Value::CreateTensor<float>(inputMemInfo(), input_ptr, input_buffer_.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    RunTimer _run_rt(device_ == "cuda");
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                                        output_names_cstr_.data(), output_names_cstr_.size());
    double _run_ms = _run_rt.elapsed_ms();
    if (bench_enabled()) {
        log_pure_inference(_run_ms);
    }

    if (output_tensors.empty() || output_tensors.size() < 2) {
        throw std::runtime_error("Instance segmentation model should have two outputs");
    }

    float *detection_array = output_tensors[0].GetTensorMutableData<float>();
    auto detection_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t detection_size = detection_tensor_info.GetElementCount();
    std::vector<int64_t> detection_output_shape = detection_tensor_info.GetShape();
    std::vector<float> detection_output;
    readOutput(detection_array, detection_size, detection_output);

    float *protos_array = output_tensors[1].GetTensorMutableData<float>();
    auto protos_tensor_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    size_t protos_size = protos_tensor_info.GetElementCount();
    std::vector<int64_t> protos_output_shape = protos_tensor_info.GetShape();
    std::vector<float> protos_output;
    readOutput(protos_array, protos_size, protos_output);

    batch_timing_.count++;
    batch_timing_.preprocess_ms += std::chrono::duration<double, std::milli>(_pre1 - _pre0).count();
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
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
        Ort::Value::CreateTensor<float>(inputMemInfo(), input_ptr, input_data.size(),
                                        input_shape_.data(), input_shape_.size());
    auto _ten1 = std::chrono::steady_clock::now();

    RunTimer _run_rt(device_ == "cuda");
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_cstr_.data(), &input_tensor, 1,
                                        output_names_cstr_.data(), output_names_cstr_.size());
    double _run_ms = _run_rt.elapsed_ms();
    if (bench_enabled()) {
        log_pure_inference(_run_ms);
    }

    if (output_tensors.empty()) {
        throw std::runtime_error("Inference output is empty");
    }

    float *float_array = output_tensors.front().GetTensorMutableData<float>();
    auto tensor_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    std::vector<float> output;
    readOutput(float_array, output_size, output);

    // 分类路径：tensor/run 在此累加；count/preprocess 由 InferSingle 累加
    batch_timing_.tensor_ms += std::chrono::duration<double, std::milli>(_ten1 - _ten0).count();
    batch_timing_.run_ms += _run_ms;
    return output;
#else
    throw std::runtime_error("ONNX Runtime not found");
#endif
}

void OnnxInference::ensureCudaInput(size_t float_count) {
#ifdef USE_CUDA
    if (float_count > cuda_input_count_) {
        if (cuda_input_) {
            cudaFree(cuda_input_);
            cuda_input_ = nullptr;
            cuda_input_count_ = 0;
        }
        if (cudaMalloc((void **)&cuda_input_, float_count * sizeof(float)) == cudaSuccess) {
            cuda_input_count_ = float_count;
        } else {
            cuda_input_ = nullptr;   // 分配失败：保持空，下次调用重试
        }
    }
#else
    (void)float_count;
#endif
}

bool OnnxInference::useGpuTensor() const {
    return device_ == "cuda" && h2d_split_enabled();
}

#ifdef ONNXRUNTIME_FOUND
const Ort::MemoryInfo &OnnxInference::inputMemInfo() const {
    return input_on_gpu_ ? cuda_mem_info_ : memory_info_;
}
#endif

float *OnnxInference::prepareInput(const float *data, size_t count) {
    // 非 split：原样返回 CPU 指针供建 CPU tensor（ORT 内部自行 H2D）。
    input_on_gpu_ = false;
    if (!useGpuTensor()) return const_cast<float *>(data);
#ifdef USE_CUDA
    if (count == 0) return const_cast<float *>(data);
    ensureCudaInput(count);
    if (!cuda_input_) return const_cast<float *>(data);   // cudaMalloc 失败回退 CPU 路径
    batch_timing_.h2d_ms += timedCudaCopyAsync(cuda_input_, data, count * sizeof(float), cudaMemcpyHostToDevice);
    batch_timing_.h2d_split = true;
    input_on_gpu_ = true;
    return cuda_input_;
#else
    return const_cast<float *>(data);
#endif
}

void OnnxInference::readOutput(const float *src, size_t count, std::vector<float> &dst) {
    dst.resize(count);
    if (count == 0) return;
    if (!input_on_gpu_) {
        // src 为 CPU 指针（非 split：ORT 内部已 D2H 到 CPU），直接拷贝
        std::copy(src, src + count, dst.begin());
        return;
    }
#ifdef USE_CUDA
    // src 为 GPU 指针（split：输入在 GPU，输出亦在 GPU），真实 D2H
    batch_timing_.d2h_ms += timedCudaCopyAsync(dst.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost);
    batch_timing_.h2d_split = true;
#else
    std::copy(src, src + count, dst.begin());
#endif
}

} // namespace inference
} // namespace JHDeepCore
