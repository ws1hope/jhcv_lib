#pragma once

#ifdef _WIN32
#ifdef PENMA_DLL_EXPORTS
#define PENMA_API __declspec(dllexport)
#else
#define PENMA_API __declspec(dllimport)
#endif
#else
#define PENMA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

PENMA_API int penma_init(
    const char* label_model,
    const char* zifu_model,
    const char* ocr_rec_model,
    const char* ocr_rec_label,
    const char* cls_model,
    int use_gpu,
    int gpu_id
);

PENMA_API int penma_recognize(
    const unsigned char* img_data,
    int width,
    int height,
    int channels,
    char* result_buf,
    int buf_size,
    const char* heat_str
);

PENMA_API int penma_recognize_file(
    const char* image_path,
    char* result_buf,
    int buf_size,
    const char* heat_str
);

PENMA_API int penma_save_result(const char* save_path);

PENMA_API void penma_destroy();

#ifdef __cplusplus
}
#endif