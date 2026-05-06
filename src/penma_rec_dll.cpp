#define PENMA_DLL_EXPORTS
#include "penma_rec_dll.h"
#include "penma_rec_inference.h"
#include <memory>
#include <iostream>
#include <cstring>

static std::unique_ptr<PenmaRecInference> g_penma_rec;
static PenmaRecResult g_last_result;

int penma_init(
    const char* label_model,
    const char* zifu_model,
    const char* ocr_rec_model,
    const char* ocr_rec_label,
    const char* cls_model,
    int use_gpu,
    int gpu_id)
{
    if (!label_model || !zifu_model || !ocr_rec_model) {
        std::cerr << "[penma_rec_dll] Missing required model paths" << std::endl;
        return -1;
    }

    try {
        PenmaRecParams params;
        params.label_model_path = std::string(label_model);
        params.zifu_model_path = std::string(zifu_model);
        params.ocr_rec_model_path = std::string(ocr_rec_model);
        params.useGPU = use_gpu != 0;
        params.gpuId = gpu_id;

        if (ocr_rec_label && strlen(ocr_rec_label) > 0) {
            params.ocr_rec_label_path = std::string(ocr_rec_label);
        }

        if (cls_model && strlen(cls_model) > 0) {
            params.cls_model_path = std::string(cls_model);
        }

        g_penma_rec = std::make_unique<PenmaRecInference>(params);
        g_penma_rec->warmup();

        std::cout << "[penma_rec_dll] Initialized successfully" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[penma_rec_dll] Init failed: " << e.what() << std::endl;
        return -2;
    }
}

int penma_recognize(
    const unsigned char* img_data,
    int width,
    int height,
    int channels,
    char* result_buf,
    int buf_size,
    const char* heat_str)
{
    if (!g_penma_rec) {
        std::cerr << "[penma_rec_dll] Not initialized" << std::endl;
        return -1;
    }

    if (!img_data || !result_buf || buf_size <= 0) {
        std::cerr << "[penma_rec_dll] Invalid input" << std::endl;
        return -2;
    }

    try {
        cv::Mat image;
        if (channels == 3) {
            image = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char*>(img_data)).clone();
        } else if (channels == 1) {
            cv::Mat gray(height, width, CV_8UC1, const_cast<unsigned char*>(img_data));
            cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);
        } else if (channels == 4) {
            cv::Mat rgba(height, width, CV_8UC4, const_cast<unsigned char*>(img_data));
            cv::cvtColor(rgba, image, cv::COLOR_BGRA2BGR);
        } else {
            std::cerr << "[penma_rec_dll] Unsupported channels: " << channels << std::endl;
            return -3;
        }

        std::string heat = heat_str ? std::string(heat_str) : "";
        g_last_result = g_penma_rec->recognize(image, heat);

        if (g_last_result.success) {
            memset(result_buf, 0, buf_size);
            size_t copy_len = std::min(g_last_result.ocr_result.size(), static_cast<size_t>(buf_size - 1));
            memcpy(result_buf, g_last_result.ocr_result.c_str(), copy_len);
            return 0;
        } else {
            result_buf[0] = '\0';
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[penma_rec_dll] Recognize failed: " << e.what() << std::endl;
        return -4;
    }
}

int penma_recognize_file(
    const char* image_path,
    char* result_buf,
    int buf_size,
    const char* heat_str)
{
    if (!g_penma_rec) {
        std::cerr << "[penma_rec_dll] Not initialized" << std::endl;
        return -1;
    }

    if (!image_path || !result_buf || buf_size <= 0) {
        std::cerr << "[penma_rec_dll] Invalid input" << std::endl;
        return -2;
    }

    try {
        cv::Mat image = cv::imread(std::string(image_path));
        if (image.empty()) {
            std::cerr << "[penma_rec_dll] Failed to load image: " << image_path << std::endl;
            return -3;
        }

        std::string heat = heat_str ? std::string(heat_str) : "";
        g_last_result = g_penma_rec->recognize(image, heat);

        if (g_last_result.success) {
            memset(result_buf, 0, buf_size);
            size_t copy_len = std::min(g_last_result.ocr_result.size(), static_cast<size_t>(buf_size - 1));
            memcpy(result_buf, g_last_result.ocr_result.c_str(), copy_len);
            return 0;
        } else {
            result_buf[0] = '\0';
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "[penma_rec_dll] Recognize failed: " << e.what() << std::endl;
        return -4;
    }
}

int penma_save_result(const char* save_path)
{
    if (!save_path) return -1;

    if (g_last_result.rotated_image.empty()) {
        std::cerr << "[penma_rec_dll] No result image to save" << std::endl;
        return -2;
    }

    if (cv::imwrite(std::string(save_path), g_last_result.rotated_image)) {
        return 0;
    }

    return -3;
}

void penma_destroy()
{
    g_penma_rec.reset();
    g_last_result = PenmaRecResult();
    std::cout << "[penma_rec_dll] Destroyed" << std::endl;
}