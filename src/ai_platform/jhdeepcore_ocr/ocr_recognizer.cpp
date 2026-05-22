#include "JHDeepCore.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <array>
#include <regex>
#include <chrono>
#include <thread>

#include <onnxruntime_cxx_api.h>

namespace JHDeepCore {

static std::vector<std::string> parseYamlCharDict(const std::string &yaml_path,
                                                   std::vector<float> &out_mean,
                                                   std::vector<float> &out_std) {
    std::vector<std::string> labels;
    std::ifstream ifs(yaml_path);
    if (!ifs.is_open()) {
        std::cerr << "[WARN] Cannot open yaml: " << yaml_path << std::endl;
        return labels;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    {
        std::regex mean_re("\"mean\"\\s*:\\s*\\[([^\\]]+)\\]");
        std::smatch m;
        if (std::regex_search(content, m, mean_re) && m.size() > 1) {
            std::string vals = m[1].str();
            out_mean.clear();
            std::istringstream iss(vals);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                out_mean.push_back(std::stof(tok));
            }
        }
    }

    {
        std::regex std_re("\"std\"\\s*:\\s*\\[([^\\]]+)\\]");
        std::smatch m;
        if (std::regex_search(content, m, std_re) && m.size() > 1) {
            std::string vals = m[1].str();
            out_std.clear();
            std::istringstream iss(vals);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                out_std.push_back(std::stof(tok));
            }
        }
    }

    {
        size_t dict_start = content.find("\"character_dict\"");
        if (dict_start != std::string::npos) {
            size_t bracket_start = content.find('[', dict_start);
            if (bracket_start != std::string::npos) {
                size_t bracket_end = content.find(']', bracket_start);
                if (bracket_end != std::string::npos) {
                    std::string dict_str = content.substr(bracket_start, bracket_end - bracket_start + 1);
                    std::regex char_re("\"((?:[^\"\\\\]|\\\\.)*)\"");
                    std::sregex_iterator it(dict_str.begin(), dict_str.end(), char_re);
                    std::sregex_iterator end;
                    for (; it != end; ++it) {
                        std::string ch = (*it)[1].str();
                        std::string decoded;
                        for (size_t i = 0; i < ch.size(); ++i) {
                            if (ch[i] == '\\' && i + 1 < ch.size()) {
                                if (ch[i + 1] == '"') { decoded += '"'; i++; }
                                else if (ch[i + 1] == '\\') { decoded += '\\'; i++; }
                                else if (ch[i + 1] == 'n') { decoded += '\n'; i++; }
                                else if (ch[i + 1] == 't') { decoded += '\t'; i++; }
                                else if (ch[i + 1] == 'U') {
                                    if (i + 10 < ch.size() && ch[i + 2] == '0' && ch[i + 3] == '0' && ch[i + 4] == '0') {
                                        std::string hex = ch.substr(i + 5, 4);
                                        unsigned int codepoint = std::stoul(hex, nullptr, 16);
                                        if (codepoint < 0x80) {
                                            decoded += static_cast<char>(codepoint);
                                        } else if (codepoint < 0x800) {
                                            decoded += static_cast<char>(0xC0 | (codepoint >> 6));
                                            decoded += static_cast<char>(0x80 | (codepoint & 0x3F));
                                        } else {
                                            decoded += static_cast<char>(0xE0 | (codepoint >> 12));
                                            decoded += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                            decoded += static_cast<char>(0x80 | (codepoint & 0x3F));
                                        }
                                        i += 10;
                                    } else {
                                        decoded += ch[i];
                                    }
                                }
                                else { decoded += ch[i]; }
                            } else {
                                decoded += ch[i];
                            }
                        }
                        labels.push_back(decoded);
                    }
                }
            }
        }
    }

    return labels;
}

static void loadRecLabels(const std::string &path, std::vector<std::string> &out_labels,
                           std::vector<float> &out_mean, std::vector<float> &out_std) {
    if (path.empty()) return;
    if (path.size() >= 5 && (path.substr(path.size() - 5) == ".yaml" || path.substr(path.size() - 5) == ".yml")) {
        out_labels = parseYamlCharDict(path, out_mean, out_std);
    } else {
        std::ifstream ifs(path);
        if (ifs.is_open()) {
            std::string line;
            while (std::getline(ifs, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                out_labels.push_back(line);
            }
        }
    }
}

static std::vector<float> preprocessRec(const cv::Mat &text_image, int rec_img_h, int rec_img_w,
                                          const std::vector<float> &mean, const std::vector<float> &std_val) {
    cv::Mat rec_resized;
    cv::resize(text_image, rec_resized, cv::Size(rec_img_w, rec_img_h));

    cv::Mat rec_blob;
    rec_resized.convertTo(rec_blob, CV_32FC3);

    bool normalized_255 = (mean.size() > 0 && mean[0] > 1.0f);

    std::vector<cv::Mat> rec_channels(3);
    cv::split(rec_blob, rec_channels);
    for (int c = 0; c < 3; ++c) {
        if (normalized_255) {
            rec_channels[c] = (rec_channels[c] - mean[c]) / std_val[c];
        } else {
            rec_channels[c] = (rec_channels[c] / 255.0f - mean[c]) / std_val[c];
        }
    }
    cv::Mat rec_normalized;
    cv::merge(rec_channels, rec_normalized);

    std::vector<float> rec_input(3 * rec_img_h * rec_img_w);
    for (int h = 0; h < rec_img_h; ++h) {
        for (int w = 0; w < rec_img_w; ++w) {
            for (int c = 0; c < 3; ++c) {
                rec_input[c * rec_img_h * rec_img_w + h * rec_img_w + w] =
                    rec_normalized.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    return rec_input;
}

static void initSession(const std::string &model_path, bool use_gpu, int gpu_id,
                        Ort::Env &env, std::unique_ptr<Ort::Session> &session,
                        std::vector<const char *> &input_names, std::vector<const char *> &output_names,
                        std::vector<std::string> &input_node_names, std::vector<std::string> &output_node_names) {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    if (use_gpu) {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = gpu_id;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "[INFO] Using CUDA GPU (device " << gpu_id << ")" << std::endl;
    } else {
        std::cout << "[INFO] Using CPU" << std::endl;
    }
#else
    std::cout << "[INFO] Using CPU (CUDA not enabled)" << std::endl;
#endif

#ifdef _WIN32
    std::wstring wideModelPath(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, wideModelPath.c_str(), sessionOptions);
#else
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    for (size_t i = 0; i < numInputNodes; i++) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_node_names.emplace_back(name.get());
    }

    size_t numOutputNodes = session->GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; i++) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_node_names.emplace_back(name.get());
    }

    input_names.clear();
    for (auto &n : input_node_names) {
        input_names.push_back(n.c_str());
    }
    output_names.clear();
    for (auto &n : output_node_names) {
        output_names.push_back(n.c_str());
    }
}

class OCRRecognizerPrivate {
  public:
    OCRRecognizerPrivate(const std::string &model_path, const std::string &label_path,
                         int device_id, const std::string &config_path, float score_threshold)
    {
        useGPU = (device_id >= 0);
        gpuId = (device_id >= 0) ? device_id : 0;
        rec_score_thresh = score_threshold;

        if (!label_path.empty()) {
            loadRecLabels(label_path, rec_labels, rec_mean, rec_std);
        }

        initSession(model_path, useGPU, gpuId,
                    rec_env, rec_session,
                    rec_input_names, rec_output_names,
                    rec_input_node_names, rec_output_node_names);
        std::cout << "[INFO] Rec model loaded: " << model_path << std::endl;

        {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> dummy(3 * 48 * 10, 0.0f);
            std::array<int64_t, 4> shape = {1, 3, 48, 10};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memInfo, dummy.data(), dummy.size(), shape.data(), shape.size());
            rec_session->Run(Ort::RunOptions{nullptr},
                rec_input_names.data(), &tensor, 1,
                rec_output_names.data(), rec_output_names.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::cout << "[INFO] Rec model warmup done (" << ms << " ms)" << std::endl;
        }

        std::cout << "[INFO] OCR Rec engine initialized." << std::endl;
    }

    void process(std::vector<cv::Mat> &images, std::vector<OCRResult> &results) {
        results.clear();
        for (auto &img : images) {
            results.push_back(recognizeSingle(img));
        }
    }

    size_t get_batch() const { return 1; }
    size_t get_input_width() const { return 10; }
    size_t get_input_height() const { return 48; }

  private:
    Ort::Env rec_env{ORT_LOGGING_LEVEL_WARNING, "ocr_rec"};
    std::unique_ptr<Ort::Session> rec_session;
    std::vector<const char *> rec_input_names;
    std::vector<const char *> rec_output_names;
    std::vector<std::string> rec_input_node_names;
    std::vector<std::string> rec_output_node_names;

    std::vector<std::string> rec_labels;
    std::vector<float> rec_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> rec_std = {0.229f, 0.224f, 0.225f};
    float rec_score_thresh = 0.5f;
    bool useGPU = false;
    int gpuId = 0;

    OCRResult recognizeSingle(const cv::Mat &text_image) {
        OCRResult result;

        float cr_ratio = static_cast<float>(text_image.cols) / text_image.rows;
        int rec_img_h = 48;
        int rec_img_w = static_cast<int>(rec_img_h * cr_ratio);
        rec_img_w = std::max(rec_img_w, 10);
        rec_img_w = std::min(rec_img_w, rec_img_h * 10);

        std::vector<float> rec_input = preprocessRec(text_image, rec_img_h, rec_img_w, rec_mean, rec_std);

        std::array<int64_t, 4> rec_input_shape = {1, 3, rec_img_h, rec_img_w};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value rec_input_tensor = Ort::Value::CreateTensor<float>(
            memInfo, rec_input.data(), rec_input.size(), rec_input_shape.data(), rec_input_shape.size());

        auto rec_outputs = rec_session->Run(
            Ort::RunOptions{nullptr},
            rec_input_names.data(), &rec_input_tensor, 1,
            rec_output_names.data(), rec_output_names.size());

        auto &rec_output = rec_outputs[0];
        auto rec_output_shape = rec_output.GetTensorTypeAndShapeInfo().GetShape();
        int rec_out_seq = static_cast<int>(rec_output_shape[1]);
        int rec_out_chars = static_cast<int>(rec_output_shape[2]);
        const float *rec_output_data = rec_output.GetTensorData<float>();

        std::string text;
        float total_score = 0.0f;
        int count = 0;
        int last_idx = 0;

        for (int t = 0; t < rec_out_seq; ++t) {
            int max_idx = 0;
            float max_val = rec_output_data[t * rec_out_chars];
            for (int c = 1; c < rec_out_chars; ++c) {
                if (rec_output_data[t * rec_out_chars + c] > max_val) {
                    max_val = rec_output_data[t * rec_out_chars + c];
                    max_idx = c;
                }
            }
            if (max_idx > 0 && max_idx != last_idx) {
                if (max_idx - 1 < static_cast<int>(rec_labels.size())) {
                    text += rec_labels[max_idx - 1];
                } else {
                    text += "?";
                }
                total_score += max_val;
                count++;
            }
            last_idx = max_idx;
        }

        float avg_score = (count > 0) ? total_score / count : 0.0f;

        if (avg_score >= rec_score_thresh) {
            OCRBox box;
            box.text = text;
            box.confidence = avg_score;
            result.boxes.push_back(std::move(box));
        }

        return result;
    }
};

OCRRecognizer::OCRRecognizer(const std::string &model_path, const std::string &label_path,
                             int device_id, const std::string &config_path,
                             float score_threshold)
    : m_pHandle(std::make_shared<OCRRecognizerPrivate>(model_path, label_path, device_id, config_path, score_threshold)) {}

OCRRecognizer::~OCRRecognizer() = default;

void OCRRecognizer::process(std::vector<cv::Mat> &images, std::vector<OCRResult> &results) {
    m_pHandle->process(images, results);
}

size_t OCRRecognizer::GetBatch() const { return m_pHandle->get_batch(); }
size_t OCRRecognizer::GetInputWidth() const { return m_pHandle->get_input_width(); }
size_t OCRRecognizer::GetInputHeight() const { return m_pHandle->get_input_height(); }

} // namespace JHDeepCore
