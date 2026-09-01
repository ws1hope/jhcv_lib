#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace JHDeepCore {
namespace Pipeline {

// 旧版 myPoint 等价：ROI 多边形顶点 / 起终点（camera_liuzhi_roi.json）
struct ReelPoint {
    int x = 0;
    int y = 0;
};

struct ReelPointD {
    double x = 0;
    double y = 0;
};

// 旧版 myRect 等价。注意：x/y 是检测框中心点坐标（旧推理输出即中心点），
// 与旧服务保持一致，勿改成左上角。
struct ReelRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int type = 0;
    int location = 0;
};

// 旧版 liuzhiInfo 等价：单个 ROI（输送线内/输送线外/集卷）的多边形与起终点
struct ReelRoiInfo {
    int liuzhi_id = -1;
    std::vector<ReelPoint> contour_points;          // 有效区域多边形顶点
    std::vector<ReelPoint> qishi_zhongzhi_points;   // 起点/终点，用于跨区域重复盘卷过滤
};

// 旧版 InputParamsReelLocation 等价（去掉固定数组，语义不变）
struct ReelRoiParams {
    std::vector<ReelPoint> inside_polygon;
    std::vector<ReelPoint> outside_polygon;
    std::vector<ReelPoint> collect_polygon;
    std::vector<ReelPoint> inside_qishi;
    std::vector<ReelPoint> outside_qishi;
};

// 旧版 OutputResultDetectGlobal 等价：单相机一次请求的检测结果
struct ReelCameraOutput {
    int camera_id = -1;
    int read_picture_flag = 0;  // 0=图像为空，1=正常
    std::string result_pic_path;
    std::vector<ReelRect> inside;    // 输送线内盘卷
    std::vector<ReelRect> outside;   // 输送线外（卸卷/抬卷）盘卷
    std::vector<ReelRect> collect;   // 集卷区域盘卷
};

// 单相机一帧输入（已解码图片）
struct ReelFrameInput {
    int camera_id = -1;
    cv::Mat image;
};

// 旧版 InferenceYOLOV8DET 的 Object 等价：rect.x/y 为中心点坐标
struct ReelDetObject {
    float prob = 0.f;
    cv::Rect rect;
    int classid = 0;
};

} // namespace Pipeline
} // namespace JHDeepCore
