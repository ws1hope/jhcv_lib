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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
// Reel id daily-sequence state, stored next to the debug log.
const char *kReelSeqFile = "JHMVDetect_reel_seq.txt";
// Upper bound of pending reel-leave events (oldest dropped when full).
const size_t kMaxLeaveEvents = 64;
// FIFO of reel-leave events; guarded by g_mutex.
std::deque<ReelLeaveEvent> g_leave_events;
// Upper bound of pending reel-birth events (oldest dropped when full).
const size_t kMaxBirthEvents = 64;
// FIFO of reel-birth events; guarded by g_mutex.
std::deque<ReelBirthEvent> g_birth_events;
// ABI guard: C# declares this struct as LayoutKind.Sequential (default Pack).
static_assert(sizeof(ReelBirthEvent) == 40,
              "ReelBirthEvent must stay 40 bytes to match the C# side");
// Last logged draw summary (only logged when it changes).
std::string g_last_draw_summary;

void SetLastErrorMessage(const std::string &msg)
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = msg;
}

// const char* overload: call sites inside the SEH wrapper of Detect_2 must
// not create std::string temporaries (objects with destructors trigger
// C2712 when the function also uses __try).
void SetLastErrorMessage(const char *msg)
{
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = msg;
}

std::string Timestamp()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tmv = {};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec, (int)ms.count());
    return std::string(buf);
}

std::string DirectoryOfFile(const std::string &path)
{
    std::error_code ec;
    return std::filesystem::path(path).parent_path().string();
}

std::string JoinPath(const std::string &dir, const std::string &name)
{
    return (std::filesystem::path(dir) / name).string();
}

bool CanWriteFile(const std::string &path)
{
    std::ofstream fout(path, std::ios::app);
    return fout.is_open();
}

// Pick a writable log path. Prefer the exe dir (easy to find), then the DLL
// dir, CWD (legacy), %TEMP%, and finally C:\ProgramData\JHMVDetect. A Windows
// service/shortcut often has CWD=C:\Windows\System32, which would silently
// lose the legacy log.
std::string PickWritableLogPath()
{
    std::vector<std::string> candidates;

#ifdef _WIN32
    char exePath[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) > 0) {
        candidates.push_back(JoinPath(DirectoryOfFile(exePath), kDebugLog));
    }

    HMODULE hModule = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&PickWritableLogPath, &hModule)) {
        char dllPath[MAX_PATH] = {0};
        if (GetModuleFileNameA(hModule, dllPath, MAX_PATH) > 0) {
            candidates.push_back(JoinPath(DirectoryOfFile(dllPath), kDebugLog));
        }
    }

    candidates.push_back(kDebugLog);  // relative to CWD (legacy fallback)

    char tempPath[MAX_PATH] = {0};
    if (GetTempPathA(MAX_PATH, tempPath) > 0) {
        candidates.push_back(JoinPath(tempPath, kDebugLog));
    }
    candidates.push_back(std::string("C:\\ProgramData\\JHMVDetect\\") + kDebugLog);
#else
    candidates.push_back(kDebugLog);
    candidates.push_back("/tmp/JHMVDetect_debug.log");
#endif

    for (const std::string &p : candidates) {
        std::error_code ec;
        std::filesystem::path parent = std::filesystem::path(p).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
        }
        if (CanWriteFile(p)) {
            return p;
        }
    }
    return kDebugLog;
}

// Chosen log path (picked once, then cached). Reused so the reel-seq state
// file lands in the same directory, which is already proven writable.
const std::string &DebugLogPath()
{
    static const std::string path = PickWritableLogPath();
    return path;
}

void WriteDebugLog(const std::string &msg)
{
    static std::mutex log_mutex;
    static bool wrote_header = false;
    std::lock_guard<std::mutex> lock(log_mutex);
    const std::string &log_path = DebugLogPath();
    std::ofstream fout(log_path, std::ios::app);
    if (!fout.is_open()) {
        return;
    }
    if (!wrote_header) {
        fout << "[" << Timestamp() << "] [log] path=" << log_path << std::endl;
        wrote_header = true;
    }
    fout << "[" << Timestamp() << "] " << msg << std::endl;
}

// Reel id persistence: "<YYYYMMDD> <seq>" next to the debug log, so a process
// restart on the same day continues numbering instead of resetting to _001.
std::string ReelSeqStatePath()
{
    std::string dir = DirectoryOfFile(DebugLogPath());
    if (dir.empty()) return std::string(kReelSeqFile);
    return JoinPath(dir, kReelSeqFile);
}

bool LoadReelSeq(std::string &date, int &seq)
{
    date.clear();
    seq = 0;
    const std::string path = ReelSeqStatePath();
    std::ifstream fin(path);
    if (!fin.is_open()) {
        WriteDebugLog("[ReelId] state file not found (new counter): " + path);
        return false;
    }
    std::string d;
    int s = 0;
    bool valid = static_cast<bool>(fin >> d >> s) && d.size() == 8 && s >= 0;
    if (valid) {
        for (char c : d) {
            if (c < '0' || c > '9') {
                valid = false;
                break;
            }
        }
    }
    if (!valid) {
        WriteDebugLog("[ReelId] state file invalid (reset counter): " + path);
        return false;
    }
    date = d;
    seq = s;
    WriteDebugLog("[ReelId] state loaded: date=" + date +
                  " seq=" + std::to_string(seq));
    return true;
}

bool SaveReelSeq(const std::string &date, int seq)
{
    const std::string path = ReelSeqStatePath();
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream fout(tmp_path, std::ios::trunc);
        if (!fout.is_open()) {
            WriteDebugLog("[ReelId] cannot open state tmp: " + tmp_path);
            return false;
        }
        fout << date << " " << seq << "\n";
        fout.flush();
        if (!fout) {
            WriteDebugLog("[ReelId] write state tmp failed: " + tmp_path);
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        // Windows rename may refuse to replace an existing file.
        std::error_code rm_ec;
        std::filesystem::remove(path, rm_ec);
        ec.clear();
        std::filesystem::rename(tmp_path, path, ec);
    }
    if (ec) {
        WriteDebugLog("[ReelId] rename state failed: " + path);
        return false;
    }
    WriteDebugLog("[ReelId] state saved: date=" + date +
                  " seq=" + std::to_string(seq));
    return true;
}

// Short pointer formatting for logs (callers must not create std::string
// temporaries inside the __try wrapper of Detect_2 -- C2712).
std::string PtrToHex(const void *p)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%p", p);
    return std::string(buf);
}

bool FileExists(const std::string &path)
{
    std::error_code ec;
    return std::filesystem::exists(path, ec) &&
           !std::filesystem::is_directory(path, ec);
}

// Resolve the legacy model name against CWD first (legacy contract), then the
// host exe directory and this DLL's directory. Sites often launch the C# app
// from a shortcut/service whose CWD is not the exe directory.
std::string ResolveModelPath()
{
    if (FileExists(kModelPath)) {
        WriteDebugLog(std::string("[Init] model found in CWD: ") + kModelPath);
        return std::string(kModelPath);
    }
    WriteDebugLog(std::string("[Init] model not in CWD: ") + kModelPath);

#ifdef _WIN32
    char exePath[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) > 0) {
        std::string p = DirectoryOfFile(exePath) + "\\" + kModelPath;
        if (FileExists(p)) {
            WriteDebugLog("[Init] model found in exe dir: " + p);
            return p;
        }
        WriteDebugLog("[Init] model not in exe dir: " + p);
    }

    HMODULE hModule = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&ResolveModelPath, &hModule)) {
        char dllPath[MAX_PATH] = {0};
        if (GetModuleFileNameA(hModule, dllPath, MAX_PATH) > 0) {
            std::string p = DirectoryOfFile(dllPath) + "\\" + kModelPath;
            if (FileExists(p)) {
                WriteDebugLog("[Init] model found in DLL dir: " + p);
                return p;
            }
            WriteDebugLog("[Init] model not in DLL dir: " + p);
        }
    }
#endif

    WriteDebugLog(std::string("[Init] model NOT FOUND, fallback to CWD path: ") +
                  kModelPath);
    return std::string(kModelPath);
}

// Logging helper called from the SEH wrapper. Keep the wrapper itself free of
// std::string temporaries (MSVC C2712).
void LogDetect2Args(unsigned char *inputImage, int nWidth, int nHeight,
                    int nBandNum,
                    OutputResultReelExportTargetTrack *output_params)
{
    WriteDebugLog("[Detect_2] args: input=" + PtrToHex(inputImage) +
                  " w=" + std::to_string(nWidth) +
                  " h=" + std::to_string(nHeight) +
                  " bands=" + std::to_string(nBandNum) +
                  " out=" + PtrToHex(output_params));
}

// Same purpose as above, but keeps the __try wrapper free of temporaries.
void LogDetect2Line(const char *msg)
{
    WriteDebugLog(msg);
}

// Create the pipeline. Caller must hold g_mutex.
bool InitPipelineLocked()
{
    std::string modelPath = ResolveModelPath();

    WriteDebugLog(std::string("[Init] build=\"") + __DATE__ + " " + __TIME__ + "\"");
    WriteDebugLog("[Init] model path = " + modelPath);
    WriteDebugLog("[Init] sizeof(reelBoxMatch)=" +
                  std::to_string(sizeof(reelBoxMatch)) +
                  " sizeof(InputParams)=" +
                  std::to_string(sizeof(InputParamsReelExportTargetTrack)) +
                  " sizeof(OutputResult)=" +
                  std::to_string(sizeof(OutputResultReelExportTargetTrack)));

    std::unique_ptr<JHDeepCore::Pipeline::ReelTrackPipeline> pipeline(
        new JHDeepCore::Pipeline::ReelTrackPipeline());

    // 新一轮 Init 清空离开事件队列与绘制摘要缓存
    g_leave_events.clear();
    g_birth_events.clear();
    g_last_draw_summary.clear();

    // Restore the persisted daily sequence before the first detect: the
    // callback saves it again on every reel birth.
    std::string seq_date;
    int seq_number = 0;
    LoadReelSeq(seq_date, seq_number);
    WriteDebugLog("[ReelId] state path = " + ReelSeqStatePath());

    WriteDebugLog("[Init] GPU init start");
    bool ok = pipeline->init(modelPath, true,
                             [](const std::string &date, int seq) {
                                 SaveReelSeq(date, seq);
                             });
    if (!ok) {
        // Do not retry CPU here. A native GPU failure can leave ORT/CUDA in a
        // corrupted state, and re-initializing in the same process may turn a
        // catchable SEH exception into an uncatchable access violation.
        WriteDebugLog("[Init] GPU init FAILED (CPU retry disabled)");
        SetLastErrorMessage(std::string("model init failed: ") + modelPath);
        return false;
    }
    pipeline->seedReelOrder(seq_date, seq_number);

    g_pipeline = pipeline.release();
    WriteDebugLog("[Init] GPU init OK");
    return true;
}

#ifdef _WIN32
static int LogInitCrash(EXCEPTION_POINTERS *ep)
{
    if (ep && ep->ExceptionRecord) {
        char buf[256];
        sprintf(buf, "!!! [Init] CRASH Code=0x%08X at=0x%p",
                (unsigned int)ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress);
        WriteDebugLog(std::string(buf));
    }
    SetLastErrorMessage("init crash caught by SEH");
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// SEH wrapper for Init: a native AV in ONNX Runtime/CUDA must not escape to
// the CLR (0xC0000005 becomes an uncatchable AccessViolationException there).
// Kept free of C++ objects to avoid C2712; the real work is in
// InitPipelineLocked().
static int InitImplSEH()
{
#ifdef _WIN32
    __try {
        return InitPipelineLocked() ? 0 : 1;
    } __except (LogInitCrash(GetExceptionInformation())) {
        return 2;
    }
#else
    return InitPipelineLocked() ? 0 : 1;
#endif
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
    WriteDebugLog("[Detect_2] buffer check: needed=" + std::to_string(needSize) +
                  " readable=" + std::to_string(readableSize));
    if (readableSize < needSize) {
        SetLastErrorMessage("inputImage readable range smaller than requested size");
        return false;
    }
    return true;
}

#else  // !_WIN32

static bool CheckInputBufferReadable(unsigned char *, int nWidth, int nHeight,
                                     int nBandNum)
{
    WriteDebugLog("[Detect_2] buffer check (non-Windows): needed=" +
                  std::to_string((size_t)nWidth * (size_t)nHeight *
                                 (size_t)nBandNum));
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

    WriteDebugLog("[Detect_2] enter Detect2Impl");
    // Log what the caller actually passed as "previous frame reel counts".
    // Birth detection depends on it; values <0 or >10 are treated as 0
    // (legacy protocol rule, same clamp as the pipeline).
    WriteDebugLog(
        "[Detect_2] input prev_exist=" +
        std::to_string(input_params.exist_region_previous_frame_boxCount) +
        " prev_leave=" +
        std::to_string(input_params.leave_region_previous_frame_boxCount));
    if (g_pipeline == nullptr) {
        WriteDebugLog("[Detect_2] pipeline is null, auto init...");
        if (!InitPipelineLocked()) {
            WriteDebugLog("[Detect_2] auto init FAILED");
            return NULL;
        }
    }

    cv::Mat srcImage = BufferToMat(inputImage, nWidth, nHeight, nBandNum);
    if (srcImage.empty()) {
        // Legacy trap (documented in reel_track_dll.h): return the caller's
        // input pointer. Callers must NOT free this pointer with MV_SDK_Free.
        WriteDebugLog("[Detect_2] BufferToMat returned empty (unsupported channel count)");
        SetLastErrorMessage("bufferToMat returned empty (unsupported channel count)");
        return inputImage;
    }
    WriteDebugLog("[Detect_2] BufferToMat: w=" + std::to_string(srcImage.cols) +
                  " h=" + std::to_string(srcImage.rows) +
                  " ch=" + std::to_string(srcImage.channels()));

    try {
        WriteDebugLog("[Detect_2] pipeline detect start");
        if (!g_pipeline->detect(srcImage, input_params, output_params)) {
            WriteDebugLog("[Detect_2] pipeline detect FAILED");
            SetLastErrorMessage("pipeline detect failed (model not initialized?)");
            return NULL;
        }
        WriteDebugLog("[Detect_2] pipeline detect OK");
    } catch (const std::exception &e) {
        std::string msg = std::string("detect exception: ") + e.what();
        SetLastErrorMessage(msg);
        WriteDebugLog(msg);
        return NULL;
    } catch (...) {
        SetLastErrorMessage("detect unknown exception");
        WriteDebugLog("[Detect_2] detect unknown exception");
        return NULL;
    }

    // Reel birth events into the bounded FIFO (one per born normal reel, so
    // two reels born in the same frame are both delivered with their own box).
    // Still holding g_mutex; Pop runs under the same lock.
    {
        std::vector<JHDeepCore::Pipeline::ReelBirthEventInfo> births =
            g_pipeline->takeBirthEvents();
        if (!births.empty()) {
            // Same clamp rule as the pipeline: counts <0 or >10 are treated
            // as 0 (legacy protocol). Logging both the raw input and the
            // current-frame count makes repeated births attributable:
            // prev_exist=0 on every birth means the caller is not feeding
            // the previous frame back.
            int prev_raw = input_params.exist_region_previous_frame_boxCount;
            int prev_used = (prev_raw < 0 || prev_raw > 10) ? 0 : prev_raw;
            int cur_exist = output_params->exist_region_current_frame_boxCount;
            long long now_ms = (long long)std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
            for (const auto &info : births) {
                ReelBirthEvent evt;
                evt.reel_id = info.reel_id;
                evt.order_number = info.order_number;
                evt.station_number = info.station_number;
                evt.x = info.x;
                evt.y = info.y;
                evt.width = info.width;
                evt.height = info.height;
                evt.birth_time_ms = now_ms;
                if (g_birth_events.size() >= kMaxBirthEvents) {
                    g_birth_events.pop_front();
                    WriteDebugLog("[ReelBirth] drop oldest (queue full, cap=64)");
                }
                g_birth_events.push_back(evt);
                WriteDebugLog(
                    "[ReelBirth] id=" + std::to_string(evt.reel_id) +
                    " order=" + std::to_string(evt.order_number) +
                    " station=" + std::to_string(evt.station_number) +
                    " center=(" + std::to_string(evt.x) + "," +
                    std::to_string(evt.y) + ") prev_exist=" +
                    std::to_string(prev_used) + " cur=" +
                    std::to_string(cur_exist) + " pending=" +
                    std::to_string(g_birth_events.size()));
            }
        }
    }

    // 盘卷离开事件入队（有界 FIFO）+ 绘制摘要（标签变化时才写日志）。
    // 注意：此处仍持有 g_mutex，Pop 由 MV_SDK_PopReelLeaveEvent 在同一把锁下取。
    {
        std::vector<JHDeepCore::Pipeline::ReelLeaveEventInfo> events =
            g_pipeline->takeLeaveEvents();
        if (!events.empty()) {
            long long now_ms = (long long)std::chrono::duration_cast<
                std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
            for (const auto &info : events) {
                ReelLeaveEvent evt;
                evt.reel_id = info.reel_id;
                evt.station_number = info.station_number;
                evt.x = info.x;
                evt.y = info.y;
                evt.width = info.width;
                evt.height = info.height;
                evt.leave_time_ms = now_ms;
                if (g_leave_events.size() >= kMaxLeaveEvents) {
                    g_leave_events.pop_front();
                    WriteDebugLog("[ReelEvent] drop oldest (queue full, cap=64)");
                }
                g_leave_events.push_back(evt);
                WriteDebugLog(
                    "[ReelEvent] leave id=" + std::to_string(evt.reel_id) +
                    " station=" + std::to_string(evt.station_number) +
                    " center=(" + std::to_string(evt.x) + "," +
                    std::to_string(evt.y) + ") pending=" +
                    std::to_string(g_leave_events.size()));
            }
        }
        const std::string &summary = g_pipeline->lastDrawSummary();
        if (!summary.empty() && summary != g_last_draw_summary) {
            g_last_draw_summary = summary;
            WriteDebugLog("[Draw] " + summary);
        }
    }

    unsigned char *outputResultImage = MatToUchar(srcImage);
    if (outputResultImage == NULL) {
        WriteDebugLog("[Detect_2] MatToUchar allocation failed");
        SetLastErrorMessage("matToUchar allocation failed");
        return NULL;
    }
    WriteDebugLog("[Detect_2] return " + PtrToHex(outputResultImage));
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
        WriteDebugLog("[Init] called");
        delete g_pipeline;  // repeated Init releases the old instance (fixes
        g_pipeline = nullptr;  // the legacy session leak)
        int rc = InitImplSEH();
        if (rc == 0) {
            WriteDebugLog("[Init] done OK");
        } else if (rc == 1) {
            WriteDebugLog("[Init] done FAILED");
        } else {
            WriteDebugLog("[Init] done CRASH");
        }
    } catch (const std::exception &e) {
        std::string msg = std::string("init exception: ") + e.what();
        SetLastErrorMessage(msg);
        WriteDebugLog(msg);
    } catch (...) {
        SetLastErrorMessage("init unknown exception");
        WriteDebugLog("[Init] init unknown exception");
    }
    return 0;
}

extern "C" REEL_TRACK_API unsigned char *REEL_TRACK_STDCALL
MV_SDK_ReelExportTargetTrack_Detect_2(
    unsigned char *inputImage, int nWidth, int nHeight, int nBandNum,
    InputParamsReelExportTargetTrack input_params,
    OutputResultReelExportTargetTrack *output_params)
{
    LogDetect2Args(inputImage, nWidth, nHeight, nBandNum, output_params);

    if (inputImage == NULL) {
        LogDetect2Line("[Detect_2] inputImage == NULL");
        SetLastErrorMessage("inputImage == NULL");
        return NULL;
    }
    if (output_params == NULL) {
        LogDetect2Line("[Detect_2] output_params == NULL");
        SetLastErrorMessage("output_params == NULL");
        return NULL;
    }
    if (nWidth <= 0 || nHeight <= 0 || nBandNum <= 0) {
        LogDetect2Line("[Detect_2] invalid nWidth/nHeight/nBandNum");
        SetLastErrorMessage("invalid nWidth/nHeight/nBandNum");
        return NULL;
    }
    if (!CheckInputBufferReadable(inputImage, nWidth, nHeight, nBandNum)) {
        LogDetect2Line("[Detect_2] input buffer readable check FAILED");
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

// Legacy multi-camera entry points declared by the C# side. The reel export
// track DLL does not implement them; return safe values instead of letting the
// C# caller fail with EntryPointNotFoundException.
extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL
MV_SDK_ReelCount_Init_test_id(int camera_id, int xmin_thresh, int xmax_thresh,
                              int ymin_thresh, int ymax_thresh)
{
    (void)xmin_thresh;
    (void)xmax_thresh;
    (void)ymin_thresh;
    (void)ymax_thresh;
    WriteDebugLog("[Stub] MV_SDK_ReelCount_Init_test_id called, camera_id=" +
                  std::to_string(camera_id));
    return 0;
}

extern "C" REEL_TRACK_API unsigned char *REEL_TRACK_STDCALL
MV_SDK_ReelLocation_Detect_test_id(int camera_id, const char *inputImageName,
                                   const void *param, void *result)
{
    (void)inputImageName;
    (void)param;
    (void)result;
    WriteDebugLog("[Stub] MV_SDK_ReelLocation_Detect_test_id called, camera_id=" +
                  std::to_string(camera_id));
    return NULL;
}

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL
MV_SDK_ReelExportTargetTrack_Destroy(void)
{
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        delete g_pipeline;
        g_pipeline = nullptr;
        g_leave_events.clear();
        g_birth_events.clear();
        g_last_draw_summary.clear();
        WriteDebugLog("[Destroy] pipeline released");
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

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_GetLastReelId(char *buf,
                                                                      int buf_size)
{
    if (buf == NULL || buf_size <= 0) return 0;
    std::string id;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_pipeline != nullptr) {
            id = g_pipeline->lastReelId();
        }
    }
    size_t n = id.copy(buf, (size_t)buf_size - 1);
    buf[n] = '\0';
    return (int)n;
}

extern "C" REEL_TRACK_API long long REEL_TRACK_STDCALL
MV_SDK_GetLastReelIdValue(void)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_pipeline == nullptr) return 0;
    return g_pipeline->lastReelIdValue();
}

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL
MV_SDK_PopReelBirthEvent(ReelBirthEvent *evt)
{
    if (evt == NULL) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_birth_events.empty()) return 0;
    *evt = g_birth_events.front();
    g_birth_events.pop_front();
    WriteDebugLog("[ReelBirth] pop id=" + std::to_string(evt->reel_id) +
                  " order=" + std::to_string(evt->order_number) +
                  " pending=" + std::to_string(g_birth_events.size()));
    return 1;
}

extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL
MV_SDK_PopReelLeaveEvent(ReelLeaveEvent *evt)
{
    if (evt == NULL) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_leave_events.empty()) return 0;
    *evt = g_leave_events.front();
    g_leave_events.pop_front();
    WriteDebugLog("[ReelEvent] pop id=" + std::to_string(evt->reel_id) +
                  " pending=" + std::to_string(g_leave_events.size()));
    return 1;
}

// Health check for periodic monitoring. Intentionally runs NO inference and
// touches NO tracking/event state, so probing here never produces births,
// leaves or log spam. Use this instead of Detect_2 for "is the model alive"
// style checks.
extern "C" REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_HealthCheck(void)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_pipeline == nullptr) {
        WriteDebugLog("[HealthCheck] pipeline not initialized");
        return 1;
    }
    if (!g_pipeline->valid()) {
        WriteDebugLog("[HealthCheck] detector invalid");
        return 2;
    }
    return 0;
}
