#include "reel_legacy_infer.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace JHDeepCore {
namespace Pipeline {

// ---------- LegacyReelDetector ----------

LegacyReelDetector::LegacyReelDetector(const std::string &model_path, bool use_gpu,
                                       int xmin_thresh, int xmax_thresh,
                                       int ymin_thresh, int ymax_thresh)
    : xmin_thresh_(xmin_thresh), xmax_thresh_(xmax_thresh),
      ymin_thresh_(ymin_thresh), ymax_thresh_(ymax_thresh)
{
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "reel-legacy-det");

    Ort::SessionOptions session_option;
#ifdef USE_CUDA
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_option;
            cuda_option.device_id = 0;
            cuda_option.arena_extend_strategy = 0;
            session_option.AppendExecutionProvider_CUDA(cuda_option);
        } catch (const std::exception &e) {
            std::cerr << "[WARN] CUDA provider unavailable, fallback to CPU: "
                      << e.what() << std::endl;
        }
    }
#endif
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_option.SetIntraOpNumThreads(1);
    session_option.SetLogSeverityLevel(3);

    try {
#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, wide_path.c_str(), session_option);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_option);
#endif
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel detector model load failed: " << e.what() << std::endl;
        return;
    }
    valid_ = true;

    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names_str_.clear();
    input_node_names_ptr_.clear();
    for (size_t i = 0; i < session_.GetInputCount(); i++) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_node_names_str_.push_back(name.get());
        input_node_names_ptr_.push_back(input_node_names_str_.back().c_str());
        // 注意：TensorTypeAndShapeInfo 借用 TypeInfo 的内存，
        // TypeInfo 必须是具名变量（旧版写法），不能用临时对象
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();
        input_h_ = static_cast<int>(dims[2]);
        input_w_ = static_cast<int>(dims[3]);
    }

    output_node_names_str_.clear();
    output_node_names_ptr_.clear();
    for (size_t i = 0; i < session_.GetOutputCount(); i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names_str_.push_back(name.get());
        output_node_names_ptr_.push_back(output_node_names_str_.back().c_str());
    }
    // TypeInfo 具名变量：TensorTypeAndShapeInfo 借用其内存（同上）
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto out_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto out_dims = out_tensor_info.GetShape();
    output_h_ = static_cast<int>(out_dims[1]);
    output_w_ = static_cast<int>(out_dims[2]);
}

bool LegacyReelDetector::detect(const cv::Mat &frame, std::vector<ReelDetObject> &objects)
{
    objects.clear();
    if (frame.empty() || !valid_) return false;

    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);

    // 旧版预处理：原图贴到左上角黑色正方形画布（非居中 letterbox），保持不变
    if (input_image_buffer_.empty() || input_image_buffer_.size() != cv::Size(_max, _max)) {
        input_image_buffer_ = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    } else {
        input_image_buffer_.setTo(cv::Scalar(0, 0, 0));
    }
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(input_image_buffer_(roi));

    float x_factor = input_image_buffer_.cols / static_cast<float>(input_w_);
    float y_factor = input_image_buffer_.rows / static_cast<float>(input_h_);

    cv::Mat blob = cv::dnn::blobFromImage(input_image_buffer_, 1 / 255.0,
                                          cv::Size(input_w_, input_h_),
                                          cv::Scalar(0, 0, 0), true, false);

    size_t tpixels = static_cast<size_t>(input_h_) * input_w_ * 3;
    std::array<int64_t, 4> input_shape_info{1, 3, input_h_, input_w_};
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, blob.ptr<float>(), tpixels,
        input_shape_info.data(), input_shape_info.size());

    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_.Run(Ort::RunOptions{nullptr},
                                   input_node_names_ptr_.data(), &input_tensor, 1,
                                   output_node_names_ptr_.data(),
                                   output_node_names_ptr_.size());
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel detector inference failed: " << e.what() << std::endl;
        return false;
    }

    const float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h_, output_w_, CV_32F, (float *)pdata);
    cv::Mat det_output = dout.t();

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    float *data = (float *)det_output.data;
    int rows = det_output.rows;
    int cols = det_output.cols;

    for (int i = 0; i < rows; i++) {
        float *row_ptr = data + i * cols;
        float max_score = -1.0f;
        int max_class_id = -1;
        for (int c = 4; c < cols; c++) {
            if (row_ptr[c] > max_score) {
                max_score = row_ptr[c];
                max_class_id = c - 4;
            }
        }
        if (max_score > 0.25f) {
            float cx = row_ptr[0];
            float cy = row_ptr[1];
            float ow = row_ptr[2];
            float oh = row_ptr[3];
            int x = static_cast<int>((cx - 0.5f * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5f * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            boxes.emplace_back(x, y, width, height);
            class_ids.push_back(max_class_id);
            confidences.push_back(max_score);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);

    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        // 旧版行为：NMS 后把 x/y 改写为中心点坐标
        int cx = boxes[index].x + boxes[index].width / 2;
        int cy = boxes[index].y + boxes[index].height / 2;
        boxes[index].x = cx;
        boxes[index].y = cy;
        if (cx > xmin_thresh_ && cx < xmax_thresh_ &&
            cy > ymin_thresh_ && cy < ymax_thresh_) {
            ReelDetObject tmp;
            tmp.classid = class_ids[index];
            tmp.prob = confidences[index];
            tmp.rect = boxes[index];
            objects.push_back(tmp);
        }
    }

    return true;
}

// ---------- LegacyReelClassifier ----------

LegacyReelClassifier::LegacyReelClassifier(const std::string &model_path, bool use_gpu)
{
    env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "reel-legacy-cls");

    Ort::SessionOptions session_option;
#ifdef USE_CUDA
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_option;
            cuda_option.device_id = 0;
            session_option.AppendExecutionProvider_CUDA(cuda_option);
        } catch (const std::exception &e) {
            std::cerr << "[WARN] CUDA provider unavailable, fallback to CPU: "
                      << e.what() << std::endl;
        }
    }
#endif
    session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_option.SetIntraOpNumThreads(1);
    session_option.SetLogSeverityLevel(3);

    try {
#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, wide_path.c_str(), session_option);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_option);
#endif
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel classifier model load failed: " << e.what() << std::endl;
        return;
    }
    valid_ = true;

    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names_str_.clear();
    input_node_names_ptr_.clear();
    for (size_t i = 0; i < session_.GetInputCount(); i++) {
        auto name = session_.GetInputNameAllocated(i, allocator);
        input_node_names_str_.push_back(name.get());
        input_node_names_ptr_.push_back(input_node_names_str_.back().c_str());
        // TypeInfo 具名变量：TensorTypeAndShapeInfo 借用其内存（同上）
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto dims = tensor_info.GetShape();
        input_h_ = static_cast<int>(dims[2]);
        input_w_ = static_cast<int>(dims[3]);
    }

    output_node_names_str_.clear();
    output_node_names_ptr_.clear();
    for (size_t i = 0; i < session_.GetOutputCount(); i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names_str_.push_back(name.get());
        output_node_names_ptr_.push_back(output_node_names_str_.back().c_str());
    }
    // TypeInfo 具名变量：TensorTypeAndShapeInfo 借用其内存（同上）
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto out_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto out_dims = out_tensor_info.GetShape();
    output_w_ = static_cast<int>(out_dims[1]);  // 输出形状 [1, num_classes]
}

LegacyReelClassifier::Result LegacyReelClassifier::detect(const cv::Mat &frame)
{
    Result res;
    if (frame.empty() || !valid_) return res;

    cv::Mat image = frame.clone();

    // 旧版预处理参数：scalefactor=1/((58+57+57)/3)、mean=(123,116,103)、swapRB
    double std_avg = (58.0 + 57.0 + 57.0) / 3.0;
    double scalefactor = 1.0 / std_avg;
    cv::Scalar mean_val = cv::Scalar(123, 116, 103);
    cv::Mat blob = cv::dnn::blobFromImage(image, scalefactor, cv::Size(input_w_, input_h_),
                                          mean_val, true, false);

    size_t tpixels = static_cast<size_t>(input_h_) * input_w_ * 3;
    std::array<int64_t, 4> input_shape_info{1, 3, input_h_, input_w_};
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, blob.ptr<float>(), tpixels,
        input_shape_info.data(), input_shape_info.size());

    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_.Run(Ort::RunOptions{nullptr},
                                   input_node_names_ptr_.data(), &input_tensor, 1,
                                   output_node_names_ptr_.data(),
                                   output_node_names_ptr_.size());
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] reel classifier inference failed: " << e.what() << std::endl;
        return res;
    }

    const float *pdata = ort_outputs[0].GetTensorMutableData<float>();
    int num_classes = output_w_;
    if (num_classes <= 0) return res;

    // softmax 后取 top1（旧版排序+softmax+top_k=1，结果等价）
    float max_val = pdata[0];
    for (int i = 1; i < num_classes; i++) {
        if (pdata[i] > max_val) max_val = pdata[i];
    }
    double sum = 0.0;
    std::vector<double> exps(num_classes);
    for (int i = 0; i < num_classes; i++) {
        exps[i] = std::exp(pdata[i] - max_val);
        sum += exps[i];
    }
    int best = 0;
    for (int i = 1; i < num_classes; i++) {
        if (exps[i] > exps[best]) best = i;
    }

    res.valid = true;
    res.class_id = best;
    res.confidence = static_cast<float>(exps[best] / sum);
    return res;
}

} // namespace Pipeline
} // namespace JHDeepCore
