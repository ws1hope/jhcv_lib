#include "tedai_juanqu_pipeline.h"
#include "reel_rules.h"

#include "json.hpp"
#include "file_utils.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

namespace JHDeepCore {
namespace Pipeline {

using json = nlohmann::json;

// 旧版结构体数组容量
static const int kMaxPanjuanCount = 50;

TedaiJuanquPipeline::TedaiJuanquPipeline(const TedaiJuanquServerConfig &config)
    : config_(config)
{
    if (!loadLiuzhiRois()) {
        std::cerr << "[ERROR] tedai_juanqu: failed to load liuzhi roi file: "
                  << config_.liuzhi_roi_file << std::endl;
    }
}

bool TedaiJuanquPipeline::loadLiuzhiRois()
{
    std::ifstream fin(config_.liuzhi_roi_file);
    if (!fin.is_open()) return false;

    json root = json::parse(fin, nullptr, false);
    if (root.is_discarded() || !root.contains("parameters")) return false;

    for (const auto &item : root["parameters"]) {
        if (!item.is_object() || !item.contains("camera_id")) continue;
        int camera_id = item["camera_id"].get<int>();

        std::vector<ReelRoiInfo> rois;
        if (item.contains("liuzhi_rois")) {
            for (const auto &roi_item : item["liuzhi_rois"]) {
                if (!roi_item.is_object()) continue;
                ReelRoiInfo info;
                if (roi_item.contains("liuzhi_id")) {
                    info.liuzhi_id = roi_item["liuzhi_id"].get<int>();
                }
                if (roi_item.contains("contour_points")) {
                    for (const auto &pt : roi_item["contour_points"]) {
                        if (!pt.is_object()) continue;
                        ReelPoint p;
                        p.x = pt.value("x", 0);
                        p.y = pt.value("y", 0);
                        if ((int)info.contour_points.size() < 20) {  // 旧版固定数组 [20]
                            info.contour_points.push_back(p);
                        }
                    }
                }
                if (roi_item.contains("qishi_zhongzhi")) {
                    for (const auto &pt : roi_item["qishi_zhongzhi"]) {
                        if (!pt.is_object()) continue;
                        ReelPoint p;
                        p.x = pt.value("x", 0);
                        p.y = pt.value("y", 0);
                        if ((int)info.qishi_zhongzhi_points.size() < 20) {
                            info.qishi_zhongzhi_points.push_back(p);
                        }
                    }
                }
                rois.push_back(info);
            }
        }
        if (rois.size() < 3) {
            std::cerr << "[WARN] tedai_juanqu: camera " << camera_id
                      << " has only " << rois.size()
                      << " liuzhi rois (need 3: inside/outside/collect)" << std::endl;
        }

        // 旧版映射：rois[0]=输送线内、rois[1]=输送线外、rois[2]=集卷
        ReelRoiParams params;
        if (rois.size() >= 1) {
            params.inside_polygon = rois[0].contour_points;
            params.inside_qishi = rois[0].qishi_zhongzhi_points;
        }
        if (rois.size() >= 2) {
            params.outside_polygon = rois[1].contour_points;
            params.outside_qishi = rois[1].qishi_zhongzhi_points;
        }
        if (rois.size() >= 3) {
            params.collect_polygon = rois[2].contour_points;
        }

        liuzhi_rois_[camera_id] = rois;
        camera_params_[camera_id] = params;
    }

    return !camera_params_.empty();
}

LegacyReelDetector *TedaiJuanquPipeline::detector(int camera_id)
{
    auto it = detectors_.find(camera_id);
    if (it != detectors_.end()) return it->second.get();

    std::string model_path;
    for (const auto &cam : config_.cameras) {
        if (cam.camera_id == camera_id) {
            model_path = cam.det_model;
            break;
        }
    }
    if (model_path.empty()) {
        std::cerr << "[ERROR] tedai_juanqu: no det model configured for camera "
                  << camera_id << std::endl;
        detectors_[camera_id] = nullptr;
        return nullptr;
    }

    auto det = std::make_unique<LegacyReelDetector>(
        model_path, config_.device == "cuda", 0, 5000, 0, 5000);
    if (!det->valid()) {
        std::cerr << "[ERROR] tedai_juanqu: camera " << camera_id
                  << " detector init failed: " << model_path << std::endl;
        detectors_[camera_id] = nullptr;
        return nullptr;
    }
    LegacyReelDetector *raw = det.get();
    detectors_[camera_id] = std::move(det);
    return raw;
}

LegacyReelClassifier *TedaiJuanquPipeline::classifier()
{
    if (classifier_loaded_) return classifier_.get();
    classifier_loaded_ = true;

    if (config_.classify_model.empty()) {
        std::cerr << "[ERROR] tedai_juanqu: classify_model not configured" << std::endl;
        return nullptr;
    }
    auto cls = std::make_unique<LegacyReelClassifier>(config_.classify_model,
                                                      config_.device == "cuda");
    if (!cls->valid()) {
        std::cerr << "[ERROR] tedai_juanqu: classifier init failed: "
                  << config_.classify_model << std::endl;
        return nullptr;
    }
    classifier_ = std::move(cls);
    return classifier_.get();
}

std::vector<ReelCameraOutput> TedaiJuanquPipeline::process(
    const std::vector<ReelFrameInput> &frames)
{
    std::vector<ReelCameraOutput> results;

    // 按 camera_id 升序处理（跨相机去重依赖：camera 3 依赖 camera 2、
    // camera 5 依赖 camera 3 的本请求结果）
    std::map<int, const ReelFrameInput *> frames_by_id;
    for (const auto &fr : frames) {
        frames_by_id[fr.camera_id] = &fr;
    }

    std::map<int, ReelCameraOutput> done;  // 本请求内已处理相机结果
    for (const auto &kv : frames_by_id) {
        int camera_id = kv.first;
        const ReelFrameInput *fr = kv.second;

        ReelCameraOutput out;
        out.camera_id = camera_id;

        auto params_it = camera_params_.find(camera_id);
        if (fr->image.empty() || params_it == camera_params_.end()) {
            out.read_picture_flag = 0;
        } else {
            cv::Mat src = fr->image.clone();  // 旧线程在副本上绘制并保存
            if (camera_id == 0) {
                detectReelExport(camera_id, src, params_it->second, out);
            } else {
                const ReelCameraOutput *prev_result = nullptr;
                const ReelRoiParams *prev_params = nullptr;
                if (camera_id == 3) {
                    auto it = done.find(2);
                    auto pit = camera_params_.find(2);
                    if (it != done.end() && pit != camera_params_.end()) {
                        prev_result = &it->second;
                        prev_params = &pit->second;
                    }
                } else if (camera_id == 5) {
                    auto it = done.find(3);
                    auto pit = camera_params_.find(3);
                    if (it != done.end() && pit != camera_params_.end()) {
                        prev_result = &it->second;
                        prev_params = &pit->second;
                    }
                }
                detectReelLocation(camera_id, src, params_it->second,
                                   prev_result, prev_params, out);
            }
            out.read_picture_flag = 1;
            out.result_pic_path = saveResultImage(camera_id, src);
        }

        done[camera_id] = out;
        results.push_back(out);
    }

    return results;
}

void TedaiJuanquPipeline::detectReelLocation(int camera_id, cv::Mat &src,
                                             const ReelRoiParams &params,
                                             const ReelCameraOutput *prev_result,
                                             const ReelRoiParams *prev_params,
                                             ReelCameraOutput &out)
{
    cv::Mat src_clone = src.clone();  // camera 5 分类用干净原图（旧版 src_img_clone）

    std::vector<ReelDetObject> vec_obj;
    LegacyReelDetector *det = detector(camera_id);
    if (!det || !det->detect(src, vec_obj)) {
        vec_obj.clear();
    }

    // 三区域多边形（Point 用于绘制，PointD 用于判定）
    std::vector<cv::Point> polygon1, polygon2, polygon3;
    std::vector<ReelPointD> inside_poly, outside_poly, collect_poly;
    for (const auto &p : params.inside_polygon) {
        inside_poly.push_back({(double)p.x, (double)p.y});
        polygon1.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.outside_polygon) {
        outside_poly.push_back({(double)p.x, (double)p.y});
        polygon2.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.collect_polygon) {
        collect_poly.push_back({(double)p.x, (double)p.y});
        polygon3.emplace_back(p.x, p.y);
    }
    if (polygon1.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon1}, true,
                      cv::Scalar(0, 0, 255), 4);
    }
    if (polygon2.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon2}, true,
                      cv::Scalar(0, 255, 255), 4);
    }
    if (polygon3.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon3}, true,
                      cv::Scalar(255, 0, 255), 4);
    }

    // camera 5：下卷 2 号工位分类（旧版在每个检测框循环内重复推理同一张裁剪图，
    // 结果等价，这里每帧推理一次）
    int classify_result_id = -1;
    if (camera_id == 5) {
        LegacyReelClassifier *cls = classifier();
        if (cls && cls->valid()) {
            cv::Rect roi(cv::Point(720, 375), cv::Point(1210, 820));
            cv::Rect image_bounds(0, 0, src_clone.cols, src_clone.rows);
            cv::Rect safe_roi = roi & image_bounds;
            cv::Mat cropped = src_clone(safe_roi);
            auto cls_result = cls->detect(cropped);
            if (cls_result.valid) {
                classify_result_id = cls_result.class_id;
                cv::putText(src, std::to_string(cls_result.class_id),
                            cv::Point(src.cols - 1000, 200),
                            cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(255, 0, 255), 2, 0);
            }
        }
    }

    // 三区域判定（testPoint 为框中心点）
    std::vector<ReelRect> inside_rects, outside_rects, collect_rects;
    for (const auto &obj : vec_obj) {
        ReelRect box;
        box.x = obj.rect.x;
        box.y = obj.rect.y;
        box.width = obj.rect.width;
        box.height = obj.rect.height;
        box.type = obj.classid;

        ReelPointD test_point{(double)obj.rect.x, (double)obj.rect.y};

        if (inside_poly.size() >= 3 && isPointInPolygon(test_point, inside_poly)) {
            if (obj.classid == 0) {
                inside_rects.push_back(box);
            }
        }
        if (outside_poly.size() >= 3 && isPointInPolygon(test_point, outside_poly)) {
            if (camera_id == 4) {
                if (obj.classid == 0 || obj.classid == 1) {
                    outside_rects.push_back(box);
                }
            } else if (camera_id == 5) {
                // 针对误检：正常盘卷 + 分类结果为 2 + 面积 > 50000
                if (obj.classid == 0 && classify_result_id == 2 &&
                    obj.rect.area() > 50000) {
                    outside_rects.push_back(box);
                }
            } else {
                outside_rects.push_back(box);
            }
        }
        if (collect_poly.size() >= 3 && isPointInPolygon(test_point, collect_poly)) {
            if (camera_id == 5) {
                if (obj.classid == 2) {
                    collect_rects.push_back(box);
                }
            } else {
                collect_rects.push_back(box);
            }
        }
    }

    int inside_count = (int)inside_rects.size();
    int outside_count = (int)outside_rects.size();
    int collect_count = (int)collect_rects.size();

    // 排序（相机专属规则）
    if (camera_id == 1) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYUp);
    } else if (camera_id == 2) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYDown);
    } else if (camera_id == 3) {
        std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpXUp);
    }

    // 跨区域重复盘卷过滤
    int guolv_flag = 0;
    int guolv_flag_outside = 0;
    if (camera_id > 1) {
        if (camera_id == 3 && prev_result && prev_params) {
            removeChongfuPanjuan(prev_result->inside, inside_rects, inside_count,
                                 prev_params->inside_qishi, params.inside_qishi,
                                 guolv_flag);
        }
        if (camera_id == 5 && prev_result && prev_params) {
            removeChongfuPanjuanBetweenInsideOutside(
                prev_result->inside, outside_rects, outside_count,
                prev_params->inside_qishi, params.outside_qishi, guolv_flag_outside);
        }
    }

    // ---- 绘制（与旧服务一致的样式） ----
    cv::Scalar color_center(0, 255, 0);
    cv::Scalar color_rec(255, 0, 0);
    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); ++i) {
        const ReelRect &detection = inside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_in_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x + box.width, box.y - 40,
                          text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + box.width, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        if (guolv_flag == 1) {
            cv::line(src, cv::Point(200, 0), cv::Point(200, 500),
                     cv::Scalar(255, 0, 0), 3);
        }
    }
    if (inside_count > 0) {
        cv::putText(src, std::to_string(inside_count),
                    cv::Point(src.cols - 200, 200),
                    cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);
    }

    cv::Scalar color_center_1(125, 255, 0);
    cv::Scalar color_rec_1(255, 255, 0);
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); ++i) {
        const ReelRect &detection = outside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_1, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_1, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_out_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_1, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    if (outside_count > 0) {
        cv::putText(src, std::to_string(outside_count),
                    cv::Point(src.cols - 200, 200),
                    cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);
    }
    if (guolv_flag_outside == 1) {
        cv::line(src, cv::Point(2000, 0), cv::Point(2000, 500),
                 cv::Scalar(0, 255, 0), 3);
    }

    cv::Scalar color_center_2(125, 255, 0);
    cv::Scalar color_rec_2(255, 0, 255);
    for (int i = 0; i < collect_count && i < (int)collect_rects.size(); ++i) {
        const ReelRect &detection = collect_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_2, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_2, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_collect_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_2, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    // 数量截断（旧结构体数组上限 50）
    if (inside_count > kMaxPanjuanCount) inside_count = kMaxPanjuanCount;
    if (outside_count > kMaxPanjuanCount) outside_count = kMaxPanjuanCount;
    if (collect_count > kMaxPanjuanCount) collect_count = kMaxPanjuanCount;

    // camera 3：桥下-转弯区域（y<525 且 x<500）透视变换到物理毫米坐标
    if (camera_id == 3) {
        const int convert_area_max_y = 525;
        const int convert_area_max_x = 500;
        std::vector<ReelRect> tmp_convert;
        std::vector<ReelRect> remaining = inside_rects;
        auto it = remaining.begin();
        while (it != remaining.end()) {
            if (it->y < convert_area_max_y && it->x < convert_area_max_x) {
                tmp_convert.push_back(*it);
                it = remaining.erase(it);
            } else {
                ++it;
            }
        }

        // 转换区域外的盘卷加偏移量，防止坐标交叉
        const int x_offset = 2400;
        const int y_offset = 7800;
        for (auto &r : remaining) {
            r.x += x_offset;
            r.y += y_offset;
        }

        const float real_width = 2400.0f;   // mm
        const float real_length = 7800.0f;  // mm
        // 标定数据：左上 -> 右上 -> 右下 -> 左下
        std::vector<cv::Point2f> calib_points;
        calib_points.push_back(cv::Point2f(83, 271));
        calib_points.push_back(cv::Point2f(215, 236));
        calib_points.push_back(cv::Point2f(513, 473));
        calib_points.push_back(cv::Point2f(142, 568));

        std::vector<ReelRect> transformed =
            transformRectsToPhysical(tmp_convert, calib_points, real_width, real_length);
        transformed.reserve(transformed.size() + remaining.size());
        transformed.insert(transformed.end(), remaining.begin(), remaining.end());
        inside_rects = transformed;
    }

    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); i++) {
        out.inside.push_back(inside_rects[i]);
    }
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); i++) {
        out.outside.push_back(outside_rects[i]);
    }
    for (int i = 0; i < collect_count && i < (int)collect_rects.size(); i++) {
        out.collect.push_back(collect_rects[i]);
    }
}

void TedaiJuanquPipeline::detectReelExport(int camera_id, cv::Mat &src,
                                           const ReelRoiParams &params,
                                           ReelCameraOutput &out)
{
    // 白色遮挡三个固定区域（旧版硬编码，防止误检盘卷中的钢卷和废钢）
    cv::Mat src_copy = src.clone();
    cv::Rect roi1(816, 493, 550, 290);
    cv::Rect roi2(722, 790, 570, 310);
    cv::Rect roi3(600, 280, 310, 200);
    cv::Rect image_bounds(0, 0, src_copy.cols, src_copy.rows);
    if ((roi1 & image_bounds) == roi1) src_copy(roi1) = cv::Scalar(255, 255, 255);
    if ((roi2 & image_bounds) == roi2) src_copy(roi2) = cv::Scalar(255, 255, 255);
    if ((roi3 & image_bounds) == roi3) src_copy(roi3) = cv::Scalar(255, 255, 255);

    std::vector<ReelDetObject> vec_obj;
    LegacyReelDetector *det = detector(camera_id);
    if (!det || !det->detect(src_copy, vec_obj)) {
        vec_obj.clear();
    }

    // 正常盘卷过滤：classid==0 且宽>50、高>30；异常盘卷 classid==1（本流程不输出）
    const int set_reel_width_thresh = 50;
    const int set_reel_height_thresh = 30;
    std::vector<ReelDetObject> output_normal;
    for (const auto &obj : vec_obj) {
        if (obj.classid == 0 && obj.rect.width > set_reel_width_thresh &&
            obj.rect.height > set_reel_height_thresh) {
            output_normal.push_back(obj);
        }
    }

    // 两区域多边形
    std::vector<cv::Point> polygon1, polygon2;
    std::vector<ReelPointD> inside_poly, outside_poly;
    for (const auto &p : params.inside_polygon) {
        inside_poly.push_back({(double)p.x, (double)p.y});
        polygon1.emplace_back(p.x, p.y);
    }
    for (const auto &p : params.outside_polygon) {
        outside_poly.push_back({(double)p.x, (double)p.y});
        polygon2.emplace_back(p.x, p.y);
    }
    if (polygon1.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon1}, true,
                      cv::Scalar(0, 0, 255), 4);
    }
    if (polygon2.size() >= 3) {
        cv::polylines(src, std::vector<std::vector<cv::Point>>{polygon2}, true,
                      cv::Scalar(0, 255, 255), 4);
    }

    std::vector<ReelRect> inside_rects, outside_rects;
    for (const auto &obj : output_normal) {
        ReelRect box;
        box.x = obj.rect.x;
        box.y = obj.rect.y;
        box.width = obj.rect.width;
        box.height = obj.rect.height;
        box.type = obj.classid;

        ReelPointD test_point{(double)obj.rect.x, (double)obj.rect.y};
        if (inside_poly.size() >= 3 && isPointInPolygon(test_point, inside_poly)) {
            if (obj.classid == 0) {
                inside_rects.push_back(box);
            }
        }
        if (outside_poly.size() >= 3 && isPointInPolygon(test_point, outside_poly)) {
            if (obj.classid == 0) {
                outside_rects.push_back(box);
            }
        }
    }

    int inside_count = (int)inside_rects.size();
    int outside_count = (int)outside_rects.size();

    std::sort(inside_rects.begin(), inside_rects.end(), reelRectCmpYUp);
    std::sort(outside_rects.begin(), outside_rects.end(), reelRectCmpYUp);

    // 绘制（旧版样式：inside 文本在框右上）
    cv::Scalar color_center(0, 255, 0);
    cv::Scalar color_rec(255, 0, 0);
    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); ++i) {
        const ReelRect &detection = inside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_in_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    cv::putText(src, std::to_string(inside_count), cv::Point(src.cols - 200, 200),
                cv::FONT_HERSHEY_DUPLEX, 5, cv::Scalar(0, 0, 255), 2, 0);

    cv::Scalar color_center_1(125, 255, 0);
    cv::Scalar color_rec_1(255, 255, 0);
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); ++i) {
        const ReelRect &detection = outside_rects[i];
        cv::Rect box;
        box.x = detection.x - detection.width / 2;
        box.y = detection.y - detection.height / 2;
        box.height = detection.height;
        box.width = detection.width;
        cv::rectangle(src, box, color_rec_1, 2);
        cv::circle(src, cv::Point(detection.x, detection.y), 5, color_center_1, -1);
        std::string class_string =
            std::to_string(camera_id) + "cam_out_" + std::to_string(i + 1);
        cv::Size text_size =
            cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_box(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(src, text_box, color_rec_1, cv::FILLED);
        cv::putText(src, class_string, cv::Point(box.x + 5, box.y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    if (inside_count > kMaxPanjuanCount) inside_count = kMaxPanjuanCount;
    if (outside_count > kMaxPanjuanCount) outside_count = kMaxPanjuanCount;

    for (int i = 0; i < inside_count && i < (int)inside_rects.size(); i++) {
        out.inside.push_back(inside_rects[i]);
    }
    for (int i = 0; i < outside_count && i < (int)outside_rects.size(); i++) {
        out.outside.push_back(outside_rects[i]);
    }
}

std::string TedaiJuanquPipeline::saveResultImage(int camera_id, const cv::Mat &image)
{
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm *lt = std::localtime(&t);

    std::string folder =
        cv::format("%s\\camera_%02d", config_.result_dir.c_str(), camera_id);
    FileHelper::ensureDirectoryExists(folder);
    std::string date_folder = cv::format("%s\\%d%02d%02d", folder.c_str(),
                                         lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday);
    FileHelper::ensureDirectoryExists(date_folder);

    std::string save_name = cv::format(
        "%s\\%d%02d%02d%02d%02d%02d%03d_%02d.jpg", date_folder.c_str(),
        lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
        lt->tm_hour, lt->tm_min, lt->tm_sec, (int)ms.count(), camera_id);

    if (!image.empty()) {
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(75);
        cv::imwrite(save_name, image, compression_params);
    }
    return save_name;
}

} // namespace Pipeline
} // namespace JHDeepCore
