// reel_track_dll.h - C ABI of JHMVDetect.dll (reel export target tracking).
//
// ABI-compatible with the legacy JHMVDetect.dll exports used by C# P/Invoke:
//   MV_SDK_ReelExportTargetTrack_Init      (model init)
//   MV_SDK_ReelExportTargetTrack_Detect_2  (per-frame detect, buffer input)
//   MV_SDK_Free                            (free returned image buffer)
// plus new additive exports:
//   MV_SDK_ReelExportTargetTrack_Destroy   (release model/session)
//   MV_SDK_GetLastError                    (last error message)
//   MV_SDK_GetLastReelId                   (last born reel id, YYYYMMDDNNN)
//   MV_SDK_GetLastReelIdValue              (same id as long long)
//   MV_SDK_PopReelLeaveEvent               (reel crossed the leave line)
//
// Struct layouts MUST NOT change (C# uses LayoutKind.Sequential,
// Pack default). All box x/y are CENTER points (legacy format).
// input_params is passed BY VALUE (1004 bytes on x64).
//
// Legacy behavior traps intentionally kept in phase 1 (do not "fix" silently):
//  1. Detect_2 returns the CALLER's inputImage pointer when the buffer cannot
//     be converted (e.g. unsupported channel count). The caller must not pass
//     that pointer to MV_SDK_Free.
//  2. 3-channel input buffers are wrapped in-place and the result is drawn on
//     them, i.e. the caller's bitmap data is modified.
//  3. Init always returns 0 (legacy contract); model load failure is only
//     reported via MV_SDK_GetLastError / stderr.

#pragma once

// Platform export macros: MSVC uses __declspec(dllexport) + __stdcall
// (legacy ABI); other compilers (mac/linux clang/gcc) use default visibility.
#if defined(_WIN32)
#ifdef REEL_TRACK_DLL_EXPORTS
#define REEL_TRACK_API __declspec(dllexport)
#else
#define REEL_TRACK_API __declspec(dllimport)
#endif
#define REEL_TRACK_STDCALL __stdcall
#else
#define REEL_TRACK_API __attribute__((visibility("default")))
#define REEL_TRACK_STDCALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct reelBoxMatch {
    int station_number;   // station the reel belongs to (1/2), -1 unknown
    int x;                // center x
    int y;                // center y
    int width;
    int height;
    int order_number;             // daily birth sequence 1..N (0 = not a new
                                  // birth this frame); see MV_SDK_GetLastReelId
    int match_state;              // 0 not matched, 1 matched
    int continuous_match_false_count;
    int previous_x;               // previous frame center x
    int previous_y;               // previous frame center y
    int previous_width;
    int previous_height;
} reelBoxMatch;  // 12 ints = 48 bytes

typedef struct InputParamsReelExportTargetTrack {
    int exist_region_previous_frame_boxCount;   // [0,10], clamped to 0 otherwise
    int leave_region_previous_frame_boxCount;   // [0,10], clamped to 0 otherwise
    reelBoxMatch boxs_exist_region_previous[10];  // left of leave-line, prev frame
    reelBoxMatch boxs_leave_region_previous[10];  // right of leave-line, prev frame
    int isReversal;   // 1 forward/static, -1 reversed
    int set_station_one_fengang_line;
    int set_station_two_fengang_line;
    int set_leave_fengang_line;
    int previous_frame_leave_near_fengang_line_right;
    int previous_frame_leave_station_number;    // 1/2, 0 unknown
    int set_detect_deviation_pixel_value;
    int set_reel_width_thresh;
    int set_reel_height_thresh;
} InputParamsReelExportTargetTrack;  // 1004 bytes

typedef struct OutputResultReelExportTargetTrack {
    int width;
    int height;
    int channels;
    int exist_region_current_frame_boxCount;   // [0,10]
    int leave_region_current_frame_boxCount;   // [0,10]
    reelBoxMatch boxs_exist_region_current[10];
    reelBoxMatch boxs_leave_region_current[10];
    int current_frame_leave_near_fengang_line_right;
    int current_frame_leave_station_number;    // 1/2, 0 unknown
    int station_one_panjuan_out_state;   // 0 none, 1 out, -1 abnormal out
    int station_two_panjuan_out_state;   // 0 none, 1 out, -1 abnormal out
    int panjuan_back_state;              // 0 no rollback, 1 rollback
    int station_one_alarm_trigger;       // edge-triggered, 1 for a single frame
    int station_two_alarm_trigger;       // edge-triggered, 1 for a single frame
} OutputResultReelExportTargetTrack;  // 1008 bytes

// One "reel crossed the leave line" event. Same reel id is reported once only
// (deduplicated inside the DLL); the queue is drained with
// MV_SDK_PopReelLeaveEvent after each Detect_2 call.
typedef struct ReelLeaveEvent {
    long long reel_id;        // 20261104007 (YYYYMMDD + daily sequence)
    int station_number;       // 1 / 2, 0 when unknown
    int x;                    // center x at the crossing frame
    int y;                    // center y
    int width;
    int height;
    long long leave_time_ms;  // Unix epoch milliseconds
} ReelLeaveEvent;

// Model init. Legacy contract: always returns 0; load failure only via
// MV_SDK_GetLastError. Model path is "panjuan_best_location.onnx": resolved
// from the process working directory first, then the host exe directory, then
// this DLL's directory. Init tries GPU first and falls back to CPU. Calling
// Init again releases the previous instance first.
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_ReelExportTargetTrack_Init(
    int xmin_thresh, int xmax_thresh, int ymin_thresh, int ymax_thresh);

// Per-frame detect. Returns a malloc'd image buffer (annotated frame) that
// MUST be released with MV_SDK_Free; NULL on error; the caller's inputImage
// pointer if the buffer cannot be converted (see trap 1 above).
REEL_TRACK_API unsigned char* REEL_TRACK_STDCALL
MV_SDK_ReelExportTargetTrack_Detect_2(
    unsigned char* inputImage, int nWidth, int nHeight, int nBandNum,
    InputParamsReelExportTargetTrack input_params,
    OutputResultReelExportTargetTrack* output_params);

// Free buffers returned by MV_SDK_ReelExportTargetTrack_Detect_2.
// Never use Marshal.FreeHGlobal / FreeCoTaskMem on those buffers.
REEL_TRACK_API void REEL_TRACK_STDCALL MV_SDK_Free(unsigned char* pBuffer);

// Legacy multi-camera entry points declared by the C# side. This DLL only
// implements the reel export track; these stubs keep P/Invoke binding safe
// (they return 0 / NULL and only write a debug-log line).
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_ReelCount_Init_test_id(
    int camera_id, int xmin_thresh, int xmax_thresh, int ymin_thresh,
    int ymax_thresh);
REEL_TRACK_API unsigned char* REEL_TRACK_STDCALL
MV_SDK_ReelLocation_Detect_test_id(int camera_id, const char* inputImageName,
                                   const void* param, void* result);

// Release model/session and reset internal alarm state (new export).
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_ReelExportTargetTrack_Destroy(void);

// Copy the last error message into buf; returns copied length (0 if none).
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_GetLastError(char* buf, int buf_size);

// Return the reel id (YYYYMMDDNNN) assigned to the most recently born normal
// reel (first frame it was located). Copies into buf; returns copied length
// (0 if no reel has been born since Init). The daily sequence is persisted
// next to the debug log, so it keeps increasing after a process restart on
// the same day.
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_GetLastReelId(char* buf, int buf_size);

// Same value as above but numeric (long long); 0 when no reel born yet.
REEL_TRACK_API long long REEL_TRACK_STDCALL MV_SDK_GetLastReelIdValue(void);

// Pop one queued "reel left the area" event (FIFO). Returns 1 and fills evt
// when an event was available, 0 when the queue is empty. Call it in a loop
// right after Detect_2; the queue holds at most 64 events and drops the
// oldest one when full.
REEL_TRACK_API int REEL_TRACK_STDCALL MV_SDK_PopReelLeaveEvent(ReelLeaveEvent* evt);

#ifdef __cplusplus
}  // extern "C"
#endif
