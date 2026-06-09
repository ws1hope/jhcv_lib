#include "ocr_inference.h"
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
#include <yaml-cpp/yaml.h>

static std::vector<std::string> parseYamlCharDict(const std::string& yaml_path,
                                                    std::vector<float>& out_mean,
                                                    std::vector<float>& out_std) {
    std::vector<std::string> labels;

    try {
        YAML::Node root = YAML::LoadFile(yaml_path);

        if (root["character_dict"] && root["character_dict"].IsSequence()) {
            for (const auto& item : root["character_dict"]) {
                labels.push_back(item.as<std::string>());
            }
        }

        if (root["mean"] && root["mean"].IsSequence()) {
            out_mean.clear();
            for (const auto& v : root["mean"]) {
                out_mean.push_back(v.as<float>());
            }
        }

        if (root["std"] && root["std"].IsSequence()) {
            out_std.clear();
            for (const auto& v : root["std"]) {
                out_std.push_back(v.as<float>());
            }
        }

        if (!labels.empty()) {
            return labels;
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "[WARN] yaml-cpp parse failed for " << yaml_path
                  << ": " << e.what() << ", trying regex fallback" << std::endl;
    }

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

    std::cout << "[INFO] Parsed yaml: " << labels.size() << " chars, mean=["
              << (out_mean.size() > 0 ? std::to_string(out_mean[0]) : "") << ", "
              << (out_mean.size() > 1 ? std::to_string(out_mean[1]) : "") << ", "
              << (out_mean.size() > 2 ? std::to_string(out_mean[2]) : "") << "], std=["
              << (out_std.size() > 0 ? std::to_string(out_std[0]) : "") << ", "
              << (out_std.size() > 1 ? std::to_string(out_std[1]) : "") << ", "
              << (out_std.size() > 2 ? std::to_string(out_std[2]) : "") << "]" << std::endl;
    return labels;
}

static void loadRecLabels(const std::string& path, std::vector<std::string>& out_labels,
                          std::vector<float>& out_mean, std::vector<float>& out_std) {
    if (path.empty()) {
        std::cerr << "[ERROR] OCR label path is empty, recognition will produce '?' for all characters" << std::endl;
        return;
    }
    if (path.size() >= 5 && (path.substr(path.size() - 5) == ".yaml" || path.substr(path.size() - 5) == ".yml")) {
        out_labels = parseYamlCharDict(path, out_mean, out_std);
    } else {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            std::cerr << "[ERROR] Cannot open OCR label file: " << path
                      << ", recognition will produce '?' for all characters" << std::endl;
            return;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            out_labels.push_back(line);
        }
    }
    if (out_labels.empty()) {
        std::cerr << "[ERROR] OCR label dictionary is empty after loading: " << path
                  << ", recognition will produce '?' for all characters" << std::endl;
    } else {
        std::cout << "[INFO] Loaded " << out_labels.size() << " recognition labels from: " << path << std::endl;
    }
}

struct OCRInference::Impl {
    std::string det_model_path;
    std::string rec_model_path;

    Ort::Env det_env{ORT_LOGGING_LEVEL_WARNING, "ocr_det"};
    std::unique_ptr<Ort::Session> det_session;
    std::vector<const char*> det_input_names;
    std::vector<const char*> det_output_names;
    std::vector<std::string> det_input_node_names;
    std::vector<std::string> det_output_node_names;

    Ort::Env rec_env{ORT_LOGGING_LEVEL_WARNING, "ocr_rec"};
    std::unique_ptr<Ort::Session> rec_session;
    std::vector<const char*> rec_input_names;
    std::vector<const char*> rec_output_names;
    std::vector<std::string> rec_input_node_names;
    std::vector<std::string> rec_output_node_names;

    std::vector<std::string> rec_labels;
    std::vector<float> rec_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> rec_std = {0.229f, 0.224f, 0.225f};
    int det_img_h = 960;
    int det_img_w = 960;
    float det_thresh = 0.3f;
    float det_box_thresh = 0.5f;
    float det_unclip_ratio = 1.5f;
    float rec_score_thresh = 0.5f;
    bool useGPU = false;
    int gpuId = 0;
};

static cv::Mat resizeForDet(const cv::Mat& img, int target_h, int target_w) {
    int h = img.rows;
    int w = img.cols;
    float ratio = std::min(static_cast<float>(target_h) / h, static_cast<float>(target_w) / w);
    int resize_h = static_cast<int>(h * ratio);
    int resize_w = static_cast<int>(w * ratio);
    resize_h = std::max(static_cast<int>(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(static_cast<int>(std::round(resize_w / 32.0) * 32), 32);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    return resized;
}

static cv::Mat getBinaryThreshold(const float* data, int h, int w, float thresh) {
    cv::Mat bit_map(h, w, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < h * w; ++i) {
        if (data[i] > thresh) {
            bit_map.data[i] = 255;
        }
    }
    return bit_map;
}

static float boxScoreFast(const cv::Mat& bitmap, const std::vector<cv::Point>& contour) {
    cv::Rect rect = cv::boundingRect(contour);
    int xmin = std::max(0, rect.x);
    int ymin = std::max(0, rect.y);
    int xmax = std::min(bitmap.cols, rect.x + rect.width);
    int ymax = std::min(bitmap.rows, rect.y + rect.height);
    if (xmax <= xmin || ymax <= ymin) return 0.0f;
    cv::Mat roi = bitmap(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    return static_cast<float>(cv::countNonZero(roi)) / ((xmax - xmin) * (ymax - ymin));
}

static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> ordered(4);
    std::vector<float> sums(4), diffs(4);
    for (int i = 0; i < 4; ++i) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x;
    }
    ordered[0] = pts[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    ordered[2] = pts[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    ordered[1] = pts[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    ordered[3] = pts[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
    return ordered;
}

static std::vector<cv::Point2f> unclipBox(const std::vector<cv::Point2f>& box, float unclip_ratio) {
    float area = 0.0f;
    int n = static_cast<int>(box.size());
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += box[i].x * box[j].y - box[j].x * box[i].y;
    }
    area = std::abs(area) / 2.0f;
    float perimeter = 0.0f;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        perimeter += std::sqrt((box[j].x - box[i].x) * (box[j].x - box[i].x) +
                               (box[j].y - box[i].y) * (box[j].y - box[i].y));
    }
    float distance = area * unclip_ratio / (perimeter + 1e-6f);

    std::vector<cv::Point2f> result;
    for (int i = 0; i < n; ++i) {
        int prev_i = (i - 1 + n) % n;
        int next_i = (i + 1) % n;
        float dx1 = box[i].x - box[prev_i].x;
        float dy1 = box[i].y - box[prev_i].y;
        float len1 = std::sqrt(dx1 * dx1 + dy1 * dy1) + 1e-6f;
        float dx2 = box[next_i].x - box[i].x;
        float dy2 = box[next_i].y - box[i].y;
        float len2 = std::sqrt(dx2 * dx2 + dy2 * dy2) + 1e-6f;
        float nx = (dy1 / len1 + dy2 / len2);
        float ny = (-dx1 / len1 - dx2 / len2);
        float norm = std::sqrt(nx * nx + ny * ny) + 1e-6f;
        nx /= norm;
        ny /= norm;
        result.emplace_back(box[i].x + nx * distance, box[i].y + ny * distance);
    }
    return result;
}

static std::vector<cv::Point2f> getBoxFromContour(const std::vector<cv::Point>& contour) {
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f vertices[4];
    rect.points(vertices);
    return std::vector<cv::Point2f>(vertices, vertices + 4);
}

static cv::Mat cropTextImg(const cv::Mat& img, const std::vector<cv::Point2f>& box) {
    auto ordered = orderPoints(box);
    float width_a = std::sqrt((ordered[3].x - ordered[0].x) * (ordered[3].x - ordered[0].x) +
                              (ordered[3].y - ordered[0].y) * (ordered[3].y - ordered[0].y));
    float width_b = std::sqrt((ordered[2].x - ordered[1].x) * (ordered[2].x - ordered[1].x) +
                              (ordered[2].y - ordered[1].y) * (ordered[2].y - ordered[1].y));
    int max_w = static_cast<int>(std::max(width_a, width_b));
    float height_a = std::sqrt((ordered[1].x - ordered[0].x) * (ordered[1].x - ordered[0].x) +
                               (ordered[1].y - ordered[0].y) * (ordered[1].y - ordered[0].y));
    float height_b = std::sqrt((ordered[2].x - ordered[3].x) * (ordered[2].x - ordered[3].x) +
                               (ordered[2].y - ordered[3].y) * (ordered[2].y - ordered[3].y));
    int max_h = static_cast<int>(std::max(height_a, height_b));

    if (max_w < 1 || max_h < 1) return {};

    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(max_w), 0),
        cv::Point2f(static_cast<float>(max_w), static_cast<float>(max_h)),
        cv::Point2f(0, static_cast<float>(max_h))
    };
    cv::Point2f src_pts[4] = {ordered[0], ordered[1], ordered[2], ordered[3]};
    cv::Mat m = cv::getAffineTransform(src_pts, dst_pts);
    cv::Mat cropped;
    cv::warpAffine(img, cropped, m, cv::Size(max_w, max_h));
    return cropped;
}

static std::vector<float> preprocessDet(const cv::Mat& img, int target_h, int target_w,
                                         const std::vector<float>& mean, const std::vector<float>& std_val) {
    cv::Mat resized = resizeForDet(img, target_h, target_w);
    cv::Mat blob;
    resized.convertTo(blob, CV_32FC3);
    blob = blob / 255.0f;

    std::vector<cv::Mat> channels(3);
    cv::split(blob, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std_val[c];
    }
    cv::Mat normalized;
    cv::merge(channels, normalized);

    int rh = resized.rows;
    int rw = resized.cols;
    std::vector<float> input_data(3 * rh * rw);
    for (int h = 0; h < rh; ++h) {
        for (int w = 0; w < rw; ++w) {
            for (int c = 0; c < 3; ++c) {
                input_data[c * rh * rw + h * rw + w] = normalized.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    return input_data;
}

static std::vector<float> preprocessRec(const cv::Mat& text_image, int rec_img_h, int rec_img_w,
                                          const std::vector<float>& mean, const std::vector<float>& std_val) {
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

static void initSession(const std::string& model_path, bool use_gpu, int gpu_id,
                        Ort::Env& env, std::unique_ptr<Ort::Session>& session,
                        std::vector<const char*>& input_names, std::vector<const char*>& output_names,
                        std::vector<std::string>& input_node_names, std::vector<std::string>& output_node_names) {
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
    for (auto& n : input_node_names) {
        input_names.push_back(n.c_str());
    }
    output_names.clear();
    for (auto& n : output_node_names) {
        output_names.push_back(n.c_str());
    }
}

OCRInference::OCRInference(const Params& params) : impl_(std::make_unique<Impl>()) {
    impl_->det_model_path = params.det_model_path;
    impl_->rec_model_path = params.rec_model_path;
    impl_->det_img_h = params.det_img_h;
    impl_->det_img_w = params.det_img_w;
    impl_->det_thresh = params.text_det_thresh;
    impl_->det_box_thresh = params.text_det_box_thresh;
    impl_->det_unclip_ratio = params.text_det_unclip_ratio;
    impl_->rec_score_thresh = params.text_rec_score_thresh;
    impl_->useGPU = (params.device == "gpu");
    impl_->gpuId = params.gpuId;

    if (params.task_mode == OCRTaskMode::DET_ONLY || params.task_mode == OCRTaskMode::DET_REC) {
        initSession(params.det_model_path, impl_->useGPU, impl_->gpuId,
                    impl_->det_env, impl_->det_session,
                    impl_->det_input_names, impl_->det_output_names,
                    impl_->det_input_node_names, impl_->det_output_node_names);
        std::cout << "[INFO] Det model loaded: " << params.det_model_path << std::endl;

        std::cout << "[INFO] Warming up det model..." << std::endl;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> dummy(3 * 32 * 32, 0.0f);
            std::array<int64_t, 4> shape = {1, 3, 32, 32};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memInfo, dummy.data(), dummy.size(), shape.data(), shape.size());
            impl_->det_session->Run(Ort::RunOptions{nullptr},
                impl_->det_input_names.data(), &tensor, 1,
                impl_->det_output_names.data(), impl_->det_output_names.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::cout << "[INFO] Det model warmup done (" << ms << " ms)" << std::endl;
        }
    }

    if (params.task_mode == OCRTaskMode::REC_ONLY || params.task_mode == OCRTaskMode::DET_REC) {
        if (!params.rec_label_path.empty()) {
            loadRecLabels(params.rec_label_path, impl_->rec_labels, impl_->rec_mean, impl_->rec_std);
        }

        initSession(params.rec_model_path, impl_->useGPU, impl_->gpuId,
                    impl_->rec_env, impl_->rec_session,
                    impl_->rec_input_names, impl_->rec_output_names,
                    impl_->rec_input_node_names, impl_->rec_output_node_names);
        std::cout << "[INFO] Rec model loaded: " << params.rec_model_path << std::endl;

        std::cout << "[INFO] Warming up rec model..." << std::endl;
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> dummy(3 * 48 * 10, 0.0f);
            std::array<int64_t, 4> shape = {1, 3, 48, 10};
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memInfo, dummy.data(), dummy.size(), shape.data(), shape.size());
            impl_->rec_session->Run(Ort::RunOptions{nullptr},
                impl_->rec_input_names.data(), &tensor, 1,
                impl_->rec_output_names.data(), impl_->rec_output_names.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::cout << "[INFO] Rec model warmup done (" << ms << " ms)" << std::endl;
        }
    }

    std::cout << "[INFO] OCR engine initialized. Mode: "
              << (params.task_mode == OCRTaskMode::DET_ONLY ? "DET_ONLY" :
                  params.task_mode == OCRTaskMode::REC_ONLY ? "REC_ONLY" : "DET_REC")
              << std::endl;
}

OCRInference::~OCRInference() = default;

void OCRInference::initDetSession() {}
void OCRInference::initRecSession() {}

OCRDetectResult OCRInference::predict(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return {};
    }
    return predict(img);
}

OCRDetectResult OCRInference::detect_only(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return {};
    }
    return detect_only(img);
}

OCRDetectResult OCRInference::detect_only(const cv::Mat& image) {
    OCRDetectResult result;

    cv::Mat img = image.clone();
    int origin_h = img.rows;
    int origin_w = img.cols;

    cv::Mat resized = resizeForDet(img, impl_->det_img_h, impl_->det_img_w);
    int resize_h = resized.rows;
    int resize_w = resized.cols;

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_val = {0.229f, 0.224f, 0.225f};
    std::vector<float> input_data = preprocessDet(img, impl_->det_img_h, impl_->det_img_w, mean, std_val);

    std::array<int64_t, 4> input_shape = {1, 3, resize_h, resize_w};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memInfo, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    auto outputs = impl_->det_session->Run(
        Ort::RunOptions{nullptr},
        impl_->det_input_names.data(), &input_tensor, 1,
        impl_->det_output_names.data(), impl_->det_output_names.size());

    auto& output = outputs[0];
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int out_h = static_cast<int>(output_shape[2]);
    int out_w = static_cast<int>(output_shape[3]);
    const float* output_data = output.GetTensorData<float>();

    std::vector<float> prob_data(out_h * out_w);
    for (int i = 0; i < out_h * out_w; ++i) {
        float src = output_data[i * 2 + 1];
        float bg = output_data[i * 2];
        prob_data[i] = std::exp(src) / (std::exp(src) + std::exp(bg) + 1e-6f);
    }

    cv::Mat binary = getBinaryThreshold(prob_data.data(), out_h, out_w, impl_->det_thresh);

    cv::Mat bit_map_resized;
    cv::resize(binary, bit_map_resized, cv::Size(origin_w, origin_h));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bit_map_resized, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        if (contour.size() < 4) continue;
        float score = boxScoreFast(binary, contour);
        if (score < impl_->det_box_thresh) continue;

        auto box_pts = getBoxFromContour(contour);
        auto unclipped = unclipBox(box_pts, impl_->det_unclip_ratio);

        cv::RotatedRect rect = cv::minAreaRect(unclipped);
        cv::Point2f vertices[4];
        rect.points(vertices);
        std::vector<cv::Point2f> final_box(vertices, vertices + 4);

        float box_w = std::max(rect.size.width, rect.size.height);
        float box_h = std::min(rect.size.width, rect.size.height);
        if (box_w < 5 || box_h < 5) continue;

        OCRDetectBox det_box;
        for (auto& pt : final_box) {
            det_box.points.emplace_back(pt.x, pt.y);
        }
        det_box.confidence = score;
        result.boxes.push_back(std::move(det_box));
    }

    std::sort(result.boxes.begin(), result.boxes.end(),
              [](const OCRDetectBox& a, const OCRDetectBox& b) {
                  float a_y = a.points.empty() ? 0 : a.points[0].second;
                  float b_y = b.points.empty() ? 0 : b.points[0].second;
                  return a_y < b_y;
              });

    return result;
}

OCRRecResult OCRInference::recognize_only(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << image_path << std::endl;
        return {};
    }
    return recognize_only(img);
}

OCRRecResult OCRInference::recognize_only(const cv::Mat& text_image) {
    OCRRecResult result;

    if (impl_->rec_labels.empty()) {
        std::cerr << "[ERROR] rec_labels is empty, OCR recognition will produce '?' for all characters. "
                  << "Check label file path and loading." << std::endl;
    }

    float cr_ratio = static_cast<float>(text_image.cols) / text_image.rows;
    int rec_img_h = 48;
    int rec_img_w = static_cast<int>(rec_img_h * cr_ratio);
    rec_img_w = std::max(rec_img_w, 10);
    rec_img_w = std::min(rec_img_w, rec_img_h * 10);

    std::vector<float> rec_input = preprocessRec(text_image, rec_img_h, rec_img_w, impl_->rec_mean, impl_->rec_std);

    std::array<int64_t, 4> rec_input_shape = {1, 3, rec_img_h, rec_img_w};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value rec_input_tensor = Ort::Value::CreateTensor<float>(
        memInfo, rec_input.data(), rec_input.size(), rec_input_shape.data(), rec_input_shape.size());

    auto rec_outputs = impl_->rec_session->Run(
        Ort::RunOptions{nullptr},
        impl_->rec_input_names.data(), &rec_input_tensor, 1,
        impl_->rec_output_names.data(), impl_->rec_output_names.size());

    auto& rec_output = rec_outputs[0];
    auto rec_output_shape = rec_output.GetTensorTypeAndShapeInfo().GetShape();
    int rec_out_seq = static_cast<int>(rec_output_shape[1]);
    int rec_out_chars = static_cast<int>(rec_output_shape[2]);
    const float* rec_output_data = rec_output.GetTensorData<float>();

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
            if (max_idx - 1 < static_cast<int>(impl_->rec_labels.size())) {
                result.text += impl_->rec_labels[max_idx - 1];
            } else {
                result.text += "?";
            }
            total_score += max_val;
            count++;
        }
        last_idx = max_idx;
    }

    result.confidence = (count > 0) ? total_score / count : 0.0f;
    return result;
}

OCRDetectResult OCRInference::predict(const cv::Mat& image) {
    OCRDetectResult final_result;

    cv::Mat img = image.clone();
    int origin_h = img.rows;
    int origin_w = img.cols;

    cv::Mat resized = resizeForDet(img, impl_->det_img_h, impl_->det_img_w);
    int resize_h = resized.rows;
    int resize_w = resized.cols;

    std::vector<float> input_data = preprocessDet(img, impl_->det_img_h, impl_->det_img_w,
        {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});

    std::array<int64_t, 4> input_shape = {1, 3, resize_h, resize_w};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memInfo, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    auto outputs = impl_->det_session->Run(
        Ort::RunOptions{nullptr},
        impl_->det_input_names.data(), &input_tensor, 1,
        impl_->det_output_names.data(), impl_->det_output_names.size());

    auto& output = outputs[0];
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int out_h = static_cast<int>(output_shape[2]);
    int out_w = static_cast<int>(output_shape[3]);
    const float* output_data = output.GetTensorData<float>();

    std::vector<float> prob_data(out_h * out_w);
    for (int i = 0; i < out_h * out_w; ++i) {
        float src = output_data[i * 2 + 1];
        float bg = output_data[i * 2];
        prob_data[i] = std::exp(src) / (std::exp(src) + std::exp(bg) + 1e-6f);
    }

    cv::Mat binary = getBinaryThreshold(prob_data.data(), out_h, out_w, impl_->det_thresh);

    cv::Mat bit_map_resized;
    cv::resize(binary, bit_map_resized, cv::Size(origin_w, origin_h));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bit_map_resized, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        if (contour.size() < 4) continue;
        float score = boxScoreFast(binary, contour);
        if (score < impl_->det_box_thresh) continue;

        auto box_pts = getBoxFromContour(contour);
        auto unclipped = unclipBox(box_pts, impl_->det_unclip_ratio);

        cv::RotatedRect rect = cv::minAreaRect(unclipped);
        cv::Point2f vertices[4];
        rect.points(vertices);
        std::vector<cv::Point2f> final_box(vertices, vertices + 4);

        float box_w = std::max(rect.size.width, rect.size.height);
        float box_h = std::min(rect.size.width, rect.size.height);
        if (box_w < 5 || box_h < 5) continue;

        cv::Mat crop_img = cropTextImg(img, final_box);
        if (crop_img.empty()) continue;

        float cr_ratio = static_cast<float>(crop_img.cols) / crop_img.rows;
        int rec_img_h = 48;
        int rec_img_w = static_cast<int>(rec_img_h * cr_ratio);
        rec_img_w = std::max(rec_img_w, 10);
        rec_img_w = std::min(rec_img_w, rec_img_h * 10);

        std::vector<float> rec_input = preprocessRec(crop_img, rec_img_h, rec_img_w, impl_->rec_mean, impl_->rec_std);

        std::array<int64_t, 4> rec_input_shape = {1, 3, rec_img_h, rec_img_w};
        Ort::Value rec_input_tensor = Ort::Value::CreateTensor<float>(
            memInfo, rec_input.data(), rec_input.size(), rec_input_shape.data(), rec_input_shape.size());

        auto rec_outputs = impl_->rec_session->Run(
            Ort::RunOptions{nullptr},
            impl_->rec_input_names.data(), &rec_input_tensor, 1,
            impl_->rec_output_names.data(), impl_->rec_output_names.size());

        auto& rec_output = rec_outputs[0];
        auto rec_output_shape = rec_output.GetTensorTypeAndShapeInfo().GetShape();
        int rec_out_seq = static_cast<int>(rec_output_shape[1]);
        int rec_out_chars = static_cast<int>(rec_output_shape[2]);
        const float* rec_output_data = rec_output.GetTensorData<float>();

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
                if (max_idx - 1 < static_cast<int>(impl_->rec_labels.size())) {
                    text += impl_->rec_labels[max_idx - 1];
                } else {
                    text += "?";
                }
                total_score += max_val;
                count++;
            }
            last_idx = max_idx;
        }

        float avg_score = (count > 0) ? total_score / count : 0.0f;
        if (avg_score < impl_->rec_score_thresh) continue;

        OCRDetectBox det_box;
        for (auto& pt : final_box) {
            det_box.points.emplace_back(pt.x, pt.y);
        }
        det_box.text = text;
        det_box.confidence = avg_score;
        final_result.boxes.push_back(std::move(det_box));
    }

    std::sort(final_result.boxes.begin(), final_result.boxes.end(),
              [](const OCRDetectBox& a, const OCRDetectBox& b) {
                  float a_y = a.points.empty() ? 0 : a.points[0].second;
                  float b_y = b.points.empty() ? 0 : b.points[0].second;
                  return a_y < b_y;
              });

    return final_result;
}
