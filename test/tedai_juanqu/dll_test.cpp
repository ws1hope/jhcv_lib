// reel_track_dll 冒烟测试：Init -> Detect_2 连续两帧（第二帧回喂第一帧输出）
// -> MV_SDK_Free -> Destroy。验证 ABI 结构体布局与内存释放路径。
// 用法: reel_track_dll_test <image.jpg>（模型 panjuan_best_location.onnx
// 需在当前目录）

#include "reel_track_dll.h"

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cstring>

typedef unsigned char byte;

int main(int argc, char *argv[])
{
    const char *image_path = argc > 1 ? argv[1] : "test.jpg";

    int ret = MV_SDK_ReelExportTargetTrack_Init(0, 5000, 0, 5000);
    printf("Init ret=%d\n", ret);

    cv::Mat img = cv::imread(image_path);
    if (img.empty() || img.channels() != 3) {
        char err[256] = {0};
        MV_SDK_GetLastError(err, sizeof(err));
        printf("[ERROR] cannot read 3-channel image %s (last error: %s)\n",
               image_path, err);
        MV_SDK_ReelExportTargetTrack_Destroy();
        return 1;
    }
    printf("image: %dx%d x%d\n", img.cols, img.rows, img.channels());

    InputParamsReelExportTargetTrack in;
    memset(&in, 0, sizeof(in));
    in.isReversal = 1;
    in.set_station_one_fengang_line = 300;
    in.set_station_two_fengang_line = 600;
    in.set_leave_fengang_line = 900;
    in.previous_frame_leave_near_fengang_line_right = 5000;
    in.set_detect_deviation_pixel_value = 7;
    in.set_reel_width_thresh = 80;
    in.set_reel_height_thresh = 50;

    OutputResultReelExportTargetTrack out;
    memset(&out, 0, sizeof(out));

    printf("sizeof(reelBoxMatch)=%zu\n", sizeof(reelBoxMatch));
    printf("sizeof(InputParams)=%zu\n", sizeof(InputParamsReelExportTargetTrack));
    printf("sizeof(OutputResult)=%zu\n", sizeof(OutputResultReelExportTargetTrack));

    for (int frame = 0; frame < 2; frame++) {
        // 注意：Detect_2 会在 3 通道输入缓冲区上原地绘制（旧行为），
        // 每帧用原始图的副本
        cv::Mat input = img.clone();

        byte *result = MV_SDK_ReelExportTargetTrack_Detect_2(
            input.data, input.cols, input.rows, input.channels(), in, &out);

        char err[256] = {0};
        MV_SDK_GetLastError(err, sizeof(err));
        printf("frame %d: result=%p last_error='%s'\n", frame, (void *)result, err);
        printf("  exist=%d leave=%d near_line_right=%d leave_station=%d "
               "back=%d out1=%d out2=%d alarm1=%d alarm2=%d\n",
               out.exist_region_current_frame_boxCount,
               out.leave_region_current_frame_boxCount,
               out.current_frame_leave_near_fengang_line_right,
               out.current_frame_leave_station_number,
               out.panjuan_back_state,
               out.station_one_panjuan_out_state,
               out.station_two_panjuan_out_state,
               out.station_one_alarm_trigger,
               out.station_two_alarm_trigger);
        for (int i = 0; i < out.exist_region_current_frame_boxCount; i++) {
            printf("  exist[%d]: station=%d center=(%d,%d) size=%dx%d\n", i,
                   out.boxs_exist_region_current[i].station_number,
                   out.boxs_exist_region_current[i].x,
                   out.boxs_exist_region_current[i].y,
                   out.boxs_exist_region_current[i].width,
                   out.boxs_exist_region_current[i].height);
        }

        if (result != NULL) {
            MV_SDK_Free(result);
        } else {
            printf("[ERROR] Detect_2 returned NULL, abort\n");
            break;
        }

        // 第二帧：把第一帧输出回喂为输入（模拟 C# 的跨帧跟踪用法）
        in.exist_region_previous_frame_boxCount = out.exist_region_current_frame_boxCount;
        in.leave_region_previous_frame_boxCount = out.leave_region_current_frame_boxCount;
        memcpy(in.boxs_exist_region_previous, out.boxs_exist_region_current,
               sizeof(out.boxs_exist_region_current));
        memcpy(in.boxs_leave_region_previous, out.boxs_leave_region_current,
               sizeof(out.boxs_leave_region_current));
        in.previous_frame_leave_near_fengang_line_right =
            out.current_frame_leave_near_fengang_line_right;
        in.previous_frame_leave_station_number =
            out.current_frame_leave_station_number;
    }

    MV_SDK_ReelExportTargetTrack_Destroy();
    printf("Destroy done\n");
    return 0;
}
