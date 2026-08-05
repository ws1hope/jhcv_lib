#pragma once

#include "jhdeepcore_inference/base_inference.h"
#include <memory>
#include <vector>

#ifdef ONNXRUNTIME_FOUND
#include <onnxruntime_cxx_api.h>
#endif

namespace JHDeepCore {
namespace inference {

class OnnxInference : public BaseInference {
  public:
    OnnxInference(const std::string &model_path, const std::string &device = "cpu",
                  const std::vector<std::string> &class_names = {}, bool warmup = true);

    ~OnnxInference() override;

    bool LoadModel() override;

    ClassificationResult InferSingle(const cv::Mat &image);
    std::vector<ClassificationResult> InferBatch(const std::vector<cv::Mat> &images);

    SegmentationResult InferSingleSegmentation(const cv::Mat &image);
    std::vector<SegmentationResult> InferBatchSegmentation(const std::vector<cv::Mat> &images);

    DetectionResult InferSingleDetection(const cv::Mat &image);
    std::vector<DetectionResult> InferBatchDetection(const std::vector<cv::Mat> &images);

    InstanceSegmentationResult InferSingleInstanceSegmentation(const cv::Mat &image);
    std::vector<InstanceSegmentationResult> InferBatchInstanceSegmentation(const std::vector<cv::Mat> &images);

    void WarmupModel(int iterations = 5);

    InferenceTiming lastBatchTiming() const override { return batch_timing_; }

  private:
#ifdef ONNXRUNTIME_FOUND
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;        // CPU allocator 描述（非 split 路径建 CPU tensor）
    // CUDA EP 的 allocator（split 路径用它 Allocate 输入 buffer，ORT 认自己的 arena 内存；
    // GetInfo() 返回的 MemoryInfo 用于建 GPU tensor，修之前 cudaMalloc 外部 buffer 被 ORT 拒致 Conv 崩）
    std::unique_ptr<Ort::Allocator> cuda_allocator_;
    // 按 input_on_gpu_ 返回建输入 tensor 用的 MemoryInfo（split: cuda_allocator_ 的 GPU info；
    // 否则 memory_info_ 的 CPU）。CreateTensor 取 const OrtMemoryInfo*。
    const OrtMemoryInfo *inputMemInfoPtr() const;
    // 执行 Run：split 用 IoBinding（BindInput GPU + BindOutput GPU，避免 ORT 把输出 D2H 回 CPU
    // 致 readOutput 的 D2H 无效并污染 kernel_stream_）；非 split 用简单 Run + host 墙钟。
    std::vector<Ort::Value> executeRun(Ort::Value &input_tensor, double &run_ms);

    std::string input_name_;
    std::vector<int64_t> input_shape_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> output_shapes_;
#endif

    bool model_loaded_;
    bool warmup_enabled_;

    // 最近一次 InferBatch* 的分段耗时（每次 InferBatch* 起点复位，InferSingle* 累加）
    InferenceTiming batch_timing_;

    // 预分配缓冲区，避免每次推理重复申请释放
    std::vector<float> input_buffer_;
    std::vector<const char *> output_names_cstr_;
    std::vector<const char *> input_names_cstr_;

    void PreprocessForOnnx(const cv::Mat &image);
    std::vector<float> RunInference(const std::vector<float> &input_data);

    // device=="cuda" && JHDEEP_H2D_SPLIT==1：输入走真实 GPU tensor 路径（H2D/run/D2H 三段
    // 分开 cudaEvent 计时，事件记在 kernel_stream_ 上）。否则走原 CPU tensor 路径（Run 内部自行
    // H2D/D2H，不拆分）。
    bool useGpuTensor() const;
    // split 路径：把普通 host 输入 staging 到 pinned buffer，再异步 H2D 到复用的 GPU 输入；
    // 非 split：原样返回 CPU 指针。
    float *prepareInput(const float *data, size_t count);
    // 结束本次输入使用。GPU 输入为成员复用缓冲，不在每次推理释放。
    void freeInput(float *ptr);
    // 单输出/多输出统一读回。多输出一次性排入同一 stream，最后只同步一次，D2H 为所有输出之和。
    void readOutput(const float *src, size_t count, std::vector<float> &dst, double &run_ms);
    void readOutputs(const std::vector<const float *> &srcs, const std::vector<size_t> &counts,
                     const std::vector<std::vector<float> *> &dsts, double &run_ms);

    bool createGpuTimingEvents();
    void destroyGpuTimingResources();
    bool ensurePinnedInput(size_t count);
    bool ensurePinnedOutput(size_t count);
    bool ensureCudaInput(size_t count);

    // user_compute_stream：split 路径让 ORT 的 CUDA EP 跑在这条 stream 上，H2D/Run/D2H 同 stream，
    // event 才能标记连续的 GPU 阶段。void* 避免 #ifdef（无 USE_CUDA 时恒 nullptr）。
    void *kernel_stream_ = nullptr;
    // H2D start/end、compute start/end、D2H start/end。以 void* 隔离公开头与 CUDA 类型。
    void *gpu_events_[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    bool gpu_timing_ready_ = false;
    bool gpu_timing_active_ = false;
    void *pinned_input_ = nullptr;
    size_t pinned_input_count_ = 0;
    void *pinned_output_ = nullptr;
    size_t pinned_output_count_ = 0;
    void *cuda_input_ = nullptr;
    size_t cuda_input_count_ = 0;
    double inference_wall_start_ms_ = 0.0;
    // 由 prepareInput 设置、CreateTensor/readOutput 读取：本次输入是否实际落在 GPU。
    // cuda_allocator_ 未就绪时 prepareInput 回退 CPU，此标志为 false，保证 ptr 与 MemoryInfo 一致。
    bool input_on_gpu_ = false;
    // 最近一次 Run 的 H2D/D2H（per-Run 打印用，prepareInput/readOutput 写入）
    double last_h2d_ms_ = 0;
    double last_d2h_ms_ = 0;
    double last_gpu_total_ms_ = 0;
};

} // namespace inference
} // namespace JHDeepCore
