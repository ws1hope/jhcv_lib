// reel_track_dll.h - C ABI of reel_track_dll.dll (reel export target tracking).
//
// ABI-compatible with the legacy JHMVDetect.dll exports used by C# P/Invoke:
//   MV_SDK_ReelExportTargetTrack_Init      (model init)
//   MV_SDK_ReelExportTargetTrack_Detect_2  (per-frame detect, buffer input)
//   MV_SDK_Free                            (free returned image buffer)
// plus new additive exports:
//   MV_SDK_ReelExportTargetTrack_Destroy   (release model/session)
//   MV_SDK_GetLastError                    (last error message)
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
#define REEL_TRACK_API __declspec(dllexport)
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
    int order_number;             // order index inside the station
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

// Model init. Legacy contract: always returns 0; load failure only via
// MV_SDK_GetLastError. Model path is "panjuan_best_location.onnx" relative
// to the process working directory (legacy behavior). Calling Init again
// releases the previous instance first.
int MV_SDK_ReelExportTargetTrack_Init(int xmin_thresh, int xmax_thresh,
                                      int ymin_thresh, int ymax_thresh);

// Per-frame detect. Returns a malloc'd image buffer (annotated frame) that
// MUST be released with MV_SDK_Free; NULL on error; the caller's inputImage
// pointer if the buffer cannot be converted (see trap 1 above).
unsigned char* MV_SDK_ReelExportTargetTrack_Detect_2(
    unsigned char* inputImage, int nWidth, int nHeight, int nBandNum,
    InputParamsReelExportTargetTrack input_params,
    OutputResultReelExportTargetTrack* output_params);

// Free buffers returned by MV_SDK_ReelExportTargetTrack_Detect_2.
// Never use Marshal.FreeHGlobal / FreeCoTaskMem on those buffers.
void MV_SDK_Free(unsigned char* pBuffer);

// Release model/session and reset internal alarm state (new export).
int MV_SDK_ReelExportTargetTrack_Destroy(void);

// Copy the last error message into buf; returns copied length (0 if none).
int MV_SDK_GetLastError(char* buf, int buf_size);

#ifdef __cplusplus
}  // extern "C"
#endif
