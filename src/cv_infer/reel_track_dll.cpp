// reel_track_dll.cpp - C ABI wrapper around JHDeepCore::Pipeline::ReelTrackPipeline.
//
// Keeps the legacy JHMVDetect.dll export names and calling convention
// (extern "C", __stdcall) so the existing C# P/Invoke code works unchanged.
// This file only does ABI adaptation, memory management and crash guards;
// all tracking logic lives in jhcv_lib (nisco_project/tedai_juanqu).
//
// Kept ASCII-only per cv_infer convention (log messages in English).

#include "reel_track_dll.h"
#include "reel_track_pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

JHDeepCore::Pipeline::ReelTrackPipeline *g_pipeline = nullptr;
std::mutex g_mutex;  // serializes Detect_2 calls (single camera, legacy contract)
std::mutex g_error_mutex;
std::string g_last_error;

// Legacy model path: relative to the process working directory.
const char *kModelPath = "panjuan_best_location.onnx";
// Legacy debug log file name (kept for site troubleshooting familiarity).
const char *kDebugLog = "JHMVDetect_debug.log";

void SetLastErrorMessage(const std::string &msg)
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = msg;
}

void WriteDebugLog(const std::string &msg)
{
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::ofstream fout(kDebugLog, std::ios::app);
    if (fout.is_open()) {
        fout << msg << std::endl;
    }
}

// Legacy bufferToMat port:
//   4 channels -> all-zero CV_8UC4 image (legacy behavior)
//   3 channels -> Mat wrapping the caller buffer IN PLACE (no copy)
//   1 channel  -> converted to BGR
//   other      -> empty Mat (legacy had undefined behavior; empty triggers the
//                 documented "return inputImage" failure path)
cv::Mat BufferToMat(unsigned char *pBuffer, int nWidth, int nHeight, int nBandNum)
{
    if (nBandNum == 4) {
        return cv::Mat::zeros(cv::Size(nWidth, nHeight), CV_8UC4);
    }
    if (nBandNum == 3) {
        return cv::Mat(nHeight, nWidth, CV_8UC3, pBuffer);
    }
    if (nBandNum == 1) {
        cv::Mat gray(nHeight, nWidth, CV_8UC1, pBuffer);
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    return cv::Mat();
}

// Legacy matToUchar port: malloc'd buffer the caller must free with MV_SDK_Free.
unsigned char *MatToUchar(const cv::Mat &img)
{
    int img_width = img.cols;
    int img_height = img.rows;
    int img_channels = img.channels();
    unsigned char *p1 = (unsigned char *)malloc(
        sizeof(unsigned char) * (size_t)img_height * img_width * img_channels);
    if (p1 == nullptr) return nullptr;
    if (img_channels == 3) {
        for (int i = 0; i < img_width * img_height * 3; i++) {
            p1[i] = img.at<cv::Vec3b>(i / (img_width * 3),
                                      (i % (img_width * 3)) / 3)[i % 3];
        }
    } else {
        // Legacy single-channel path (multi-channel non-3 is not used by C#)
        for (int j = 0; j < img_height; j++) {
            const unsigned char *row = img.ptr<unsigned char>(j);
            for (int i = 0; i < img_width; i++) {
                int bias = j * img_width + i;
                p1[bias] = row[i];
            }
        }
    }
    return p1;
}

#ifdef _WIN32

// Crash details: read/write direction and target address, to distinguish
// out-of-bounds reads from heap corruption (legacy LogCrashDetails port).
static int LogCrashDetails(EXCEPTION_POINTERS *ep)
{
    if (ep && ep->ExceptionRecord) {
        EXCEPTION_RECORD *er = ep->ExceptionRecord;
        const char *rw = (er->ExceptionInformation[0] == 0)
                             ? "READ"
                             : ((er->ExceptionInformation[0] == 1) ? "WRITE" : "EXEC");
        char buf[512];
        sprintf(buf,
                "!!! [Detect_2] CRASH detail: Code=0x%08X at=0x%p type=%s addr=0x%llX",
                (unsigned int)er->ExceptionCode, er->ExceptionAddress, rw,
                (unsigned long long)er->ExceptionInformation[1]);
        WriteDebugLog(std::string(buf));
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

static void LogCrashException(DWORD exceptionCode)
{
    char buf[256];
    sprintf(buf, "!!! [Detect_2] CRASH caught: ExceptionCode=0x%08X",
            (unsigned int)exceptionCode);
    WriteDebugLog(std::string(buf));
}

// Returns how many bytes starting at p are actually readable (VirtualQuery walk).
static size_t GetReadableSize(const void *p, size_t needSize)
{
    const unsigned char *cur = (const unsigned char *)p;
    const unsigned char *end = cur + needSize;
    MEMORY_BASIC_INFORMATION mbi;
    size_t readable = 0;
    while (cur < end) {
        if (VirtualQuery(cur, &mbi, sizeof(mbi)) == 0) break;
        if (mbi.State != MEM_COMMIT) break;
        if ((mbi.Protect & 0xFF) == PAGE_NOACCESS || (mbi.Protect & PAGE_GUARD)) break;
        const unsigned char *regionEnd =
            (const unsigned char *)mbi.BaseAddress + mbi.RegionSize;
        if (regionEnd >= end) {
            readable = needSize;
            break;
        }
        readable += (size_t)(regionEnd - cur);
        cur = regionEnd;
    }
    return readable;
}

static bool CheckInputBufferReadable(unsigned char *inputImage, int nWidth,
                                     int nHeight, int nBandNum)
{
    size_t needSize = (size_t)nWidth * (size_t)nHeight * (size_t)nBandNum;
    size_t readableSize = GetReadableSize(inputImage, needSize);
    if (readableSize < needSize) {
        SetLastErrorMessage("inputImage readable range smaller than requested size");
        return false;
    }
    return true;
}

#else  // !_WIN32

static bool CheckInputBufferReadable(unsigned char *, int, int, int)
{
    return true;  // no VirtualQuery outside Windows
}

#endif  // _WIN32

// C++ body of Detect_2 (separate function: SEH wrapper must not contain
// objects requiring unwinding, legacy workaround for C2712).
static unsigned char *Detect2Impl(unsigned char *inputImage, int nWidth,
                                  int nHeight, int nBandNum,
                                  const InputParamsReelExportTargetTrack &input_params,
                                  OutputResultReelExportTargetTrack *output_params)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_pipeline == nullptr) {
        SetLastErrorMessage("pipeline not initialized (call Init first)");
        return NULL;
    }

    cv::Mat srcImage = BufferToMat(inputImage, nWidth, nHeight, nBandNum);
    if (srcImage.empty()) {
        // Legacy trap (documented in reel_track_dll.h): return the caller's
        // input pointer. Callers must NOT free this pointer with MV_SDK_Free.
        SetLastErrorMessage("bufferToMat returned empty (unsupported channel count)");
        return inputImage;
    }

    try {
        if (!g_pipeline->detect(srcImage, input_params, output_params)) {
            SetLastErrorMessage("pipeline detect failed (model not initialized?)");
            return NULL;
        }
    } catch (const std::exception &e) {
        SetLastErrorMessage(std::string("detect exception: ") + e.what());
        return NULL;
    } catch (...) {
        SetLastErrorMessage("detect unknown exception");
        return NULL;
    }

    unsigned char *outputResultImage = MatToUchar(srcImage);
    if (outputResultImage == NULL) {
        SetLastErrorMessage("matToUchar allocation failed");
        return NULL;
    }
    return outputResultImage;
}

}  // namespace

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_ReelExportTargetTrack_Init(
    int xmin_thresh, int xmax_thresh, int ymin_thresh, int ymax_thresh)
{
    // Legacy contract: always returns 0. Load failure is reported via
    // MV_SDK_GetLastError and Detect_2 returning NULL.
    // The four threshold params are kept for ABI compatibility only: the legacy
    // area filter (always 0..5000, a no-op) no longer exists on ai_platform.
    (void)xmin_thresh;
    (void)xmax_thresh;
    (void)ymin_thresh;
    (void)ymax_thresh;
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        delete g_pipeline;  // repeated Init releases the old instance (fixes
        g_pipeline = nullptr;  // the legacy session leak)
        std::unique_ptr<JHDeepCore::Pipeline::ReelTrackPipeline> pipeline(
            new JHDeepCore::Pipeline::ReelTrackPipeline());
        if (!pipeline->init(kModelPath, true)) {
            SetLastErrorMessage(std::string("model init failed: ") + kModelPath);
            return 0;
        }
        g_pipeline = pipeline.release();
    } catch (const std::exception &e) {
        SetLastErrorMessage(std::string("init exception: ") + e.what());
    } catch (...) {
        SetLastErrorMessage("init unknown exception");
    }
    return 0;
}

extern "C" REEL_TRACK_API unsigned char *REEL_TRACK_STDCALL
MV_SDK_ReelExportTargetTrack_Detect_2(
    unsigned char *inputImage, int nWidth, int nHeight, int nBandNum,
    InputParamsReelExportTargetTrack input_params,
    OutputResultReelExportTargetTrack *output_params)
{
    if (inputImage == NULL) {
        SetLastErrorMessage("inputImage == NULL");
        return NULL;
    }
    if (output_params == NULL) {
        SetLastErrorMessage("output_params == NULL");
        return NULL;
    }
    if (nWidth <= 0 || nHeight <= 0 || nBandNum <= 0) {
        SetLastErrorMessage("invalid nWidth/nHeight/nBandNum");
        return NULL;
    }
    if (!CheckInputBufferReadable(inputImage, nWidth, nHeight, nBandNum)) {
        return NULL;
    }

#ifdef _WIN32
    __try {
        return Detect2Impl(inputImage, nWidth, nHeight, nBandNum, input_params,
                           output_params);
    } __except (LogCrashDetails(GetExceptionInformation())) {
        LogCrashException(GetExceptionCode());
        SetLastErrorMessage("crash caught by SEH in Detect_2");
        return NULL;
    }
#else
    return Detect2Impl(inputImage, nWidth, nHeight, nBandNum, input_params,
                       output_params);
#endif
}

extern "C" REEL_TRACK_API void REEL_TRACK_STDCALL MV_SDK_Free(unsigned char *pBuffer)
{
    if (pBuffer != NULL) {
        free(pBuffer);
    }
}

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL
MV_SDK_ReelExportTargetTrack_Destroy(void)
{
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        delete g_pipeline;
        g_pipeline = nullptr;
    } catch (...) {
        // never let an exception cross the DLL boundary
    }
    return 0;
}

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_GetLastError(char *buf,
                                                                   int buf_size)
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    if (buf == NULL || buf_size <= 0) return 0;
    size_t n = g_last_error.copy(buf, (size_t)buf_size - 1);
    buf[n] = '\0';
    return (int)n;
}
