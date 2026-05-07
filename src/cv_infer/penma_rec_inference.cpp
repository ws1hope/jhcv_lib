#include "penma_rec_inference.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <cmath>

PenmaRecInference::PenmaRecInference(const PenmaRecParams& params)
    : m_params(params)
{
    InferenceConfig label_config;
    label_config.modelPath = params.label_model_path;
    label_config.useGPU = params.useGPU;
    label_config.gpuId = params.gpuId;
    label_config.confThreshold = params.label_conf_threshold;
    label_config.nmsThreshold = params.label_nms_threshold;
    label_config.inputWidth = params.yolo_input_width;
    label_config.inputHeight = params.yolo_input_height;
    label_config.classNames = {"circle"};
    m_label_detector = std::make_unique<YOLOInference>(label_config);
    std::cout << "[PenmaRec] Label model loaded: " << params.label_model_path << std::endl;

    InferenceConfig zifu_config;
    zifu_config.modelPath = params.zifu_model_path;
    zifu_config.useGPU = params.useGPU;
    zifu_config.gpuId = params.gpuId;
    zifu_config.confThreshold = params.zifu_conf_threshold;
    zifu_config.nmsThreshold = params.zifu_nms_threshold;
    zifu_config.inputWidth = params.yolo_input_width;
    zifu_config.inputHeight = params.yolo_input_height;
    zifu_config.classNames = {"zifu"};
    m_zifu_detector = std::make_unique<YOLOInference>(zifu_config);
    std::cout << "[PenmaRec] Zifu model loaded: " << params.zifu_model_path << std::endl;

    OCRInference::Params ocr_params;
    ocr_params.rec_model_path = params.ocr_rec_model_path;
    ocr_params.rec_label_path = params.ocr_rec_label_path;
    ocr_params.useGPU = params.useGPU;
    ocr_params.gpuId = params.gpuId;
    ocr_params.task_mode = OCRTaskMode::REC_ONLY;
    m_ocr = std::make_unique<OCRInference>(ocr_params);
    std::cout << "[PenmaRec] OCR engine initialized (rec only)" << std::endl;

    if (!params.cls_model_path.empty()) {
        ClsConfig cls_config;
        cls_config.modelPath = params.cls_model_path;
        cls_config.useGPU = params.useGPU;
        cls_config.gpuId = params.gpuId;
        cls_config.inputWidth = params.cls_input_width;
        cls_config.inputHeight = params.cls_input_height;
        m_cls = std::make_unique<ResNetInference>(cls_config);
        std::cout << "[PenmaRec] Angle classifier loaded: " << params.cls_model_path << std::endl;
    } else {
        std::cout << "[PenmaRec] No angle classifier, will use heuristic" << std::endl;
    }
}

PenmaRecInference::~PenmaRecInference() = default;

void PenmaRecInference::warmup()
{
    std::cout << "[PenmaRec] Warming up models..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat dummy_yolo(m_params.yolo_input_height, m_params.yolo_input_width, CV_8UC3, cv::Scalar(0, 0, 0));

    m_label_detector->detect(dummy_yolo);
    std::cout << "[PenmaRec]   Label model warmup done" << std::endl;

    m_zifu_detector->detect(dummy_yolo);
    std::cout << "[PenmaRec]   Zifu model warmup done" << std::endl;

    cv::Mat dummy_ocr(48, 192, CV_8UC3, cv::Scalar(0, 0, 0));
    m_ocr->recognize_only(dummy_ocr);
    std::cout << "[PenmaRec]   OCR model warmup done" << std::endl;

    if (m_cls) {
        cv::Mat dummy_cls(m_params.cls_input_height, m_params.cls_input_width, CV_8UC3, cv::Scalar(0, 0, 0));
        m_cls->classify(dummy_cls);
        std::cout << "[PenmaRec]   Cls model warmup done" << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[PenmaRec] Warmup complete (" << ms << " ms)" << std::endl;
}

PenmaRecResult PenmaRecInference::recognize(const cv::Mat& image, const std::string& heat_str)
{
    PenmaRecResult result;
    if (!image.data) {
        std::cerr << "[PenmaRec] Empty image" << std::endl;
        return result;
    }

    result.annotated_image = image.clone();

    auto labels = detectLabels(image);
    if (labels.empty()) {
        std::cerr << "[PenmaRec] No labels detected" << std::endl;
        return result;
    }
    result.labels = labels;

    bool any_success = false;
    std::vector<std::string> ocr_parts;

    for (size_t kk = 0; kk < labels.size(); kk++) {
        const auto& label = labels[kk];

        cv::Mat rotated_img;
        auto angle_params = detectChars(label.roi_image, rotated_img);
        if (angle_params.empty()) {
            std::cerr << "[PenmaRec] No chars detected for label " << kk << std::endl;
            continue;
        }

        cv::Mat result_0 = rotated_img.clone();

        int bias_1h = 30;
        int bias_1w = 30;
        cv::Mat img_copy2 = cv::Mat::zeros(
            label.roi_image.rows + 2 * bias_1h,
            label.roi_image.cols + 2 * bias_1w, CV_8UC3);
        cv::Rect roi_rect0(bias_1w, bias_1h, label.roi_image.cols, label.roi_image.rows);
        label.roi_image.copyTo(img_copy2(roi_rect0));

        bubbleSort_height(angle_params);
        float angle = angle_params[angle_params.size() - 1].angle;

        int sum_x = 0, sum_y = 0;
        for (const auto& ap : angle_params) {
            sum_x += ap.ocrCenterX + ap.left_top.x;
            sum_y += ap.ocrCenterY + ap.left_top.y;
        }
        int center_x = sum_x / static_cast<int>(angle_params.size());
        int center_y = sum_y / static_cast<int>(angle_params.size());

        int xOffset, yOffset;
        result_0 = Rotate(img_copy2, center_x + bias_1w, center_y + bias_1h, angle, xOffset, yOffset);

        std::vector<cv::Mat> char_images;
        std::vector<cv::Point2i> char_centers;
        std::vector<zifu_center_info> zifu_center_list;

        cv::Point2f guige_min_xy, guige_max_xy, second_zifu_min_xy, second_zifu_max_xy;
        cv::Point2f third_zifu_min_xy, third_zifu_max_xy;
        cv::Point2f paihao_min_xy, paihao_max_xy, paihao_next_zifu_min_xy, paihao_next_zifu_max_xy;
        float guige_center_y = 0, second_zifu_center_y = 0, third_zifu_center_y = 0;

        for (size_t j = 0; j < angle_params.size(); j++) {
            cv::Point2f point[4];
            point[0].x = angle_params[j].p0.x + angle_params[j].left_top.x + bias_1w;
            point[0].y = angle_params[j].p0.y + angle_params[j].left_top.y + bias_1h;
            point[1].x = angle_params[j].p1.x + angle_params[j].left_top.x + bias_1w;
            point[1].y = angle_params[j].p1.y + angle_params[j].left_top.y + bias_1h;
            point[2].x = angle_params[j].p2.x + angle_params[j].left_top.x + bias_1w;
            point[2].y = angle_params[j].p2.y + angle_params[j].left_top.y + bias_1h;
            point[3].x = angle_params[j].p3.x + angle_params[j].left_top.x + bias_1w;
            point[3].y = angle_params[j].p3.y + angle_params[j].left_top.y + bias_1h;

            std::vector<float> vec_points_x, vec_points_y;
            cv::Point2f centerP(center_x + bias_1w, center_y + bias_1h);
            for (int k = 0; k < 4; k++) {
                cv::Point2f middle = rotatePoint(point[k], centerP, -angle);
                vec_points_x.push_back(middle.x + xOffset);
                vec_points_y.push_back(middle.y + yOffset);
            }

            int max_x = static_cast<int>(*std::max_element(vec_points_x.begin(), vec_points_x.end()));
            int max_y = static_cast<int>(*std::max_element(vec_points_y.begin(), vec_points_y.end()));
            int min_x = static_cast<int>(*std::min_element(vec_points_x.begin(), vec_points_x.end()));
            int min_y = static_cast<int>(*std::min_element(vec_points_y.begin(), vec_points_y.end()));

            cv::Rect roi_rect = safeROI(min_x, min_y, max_x - min_x, max_y - min_y, result_0.cols, result_0.rows);
            if (roi_rect.width <= 0 || roi_rect.height <= 0) continue;

            cv::Mat roi = result_0(roi_rect);
            if (roi.empty()) continue;

            if (j == 0) {
                guige_min_xy = cv::Point2f(min_x, min_y);
                guige_max_xy = cv::Point2f(max_x, max_y);
                guige_center_y = (max_y + min_y) / 2.0f;
                zifu_center_list.push_back({guige_min_xy, guige_max_xy, guige_center_y});
            } else if (j == 1) {
                second_zifu_min_xy = cv::Point2f(min_x, min_y);
                second_zifu_max_xy = cv::Point2f(max_x, max_y);
                second_zifu_center_y = (max_y + min_y) / 2.0f;
                zifu_center_list.push_back({second_zifu_min_xy, second_zifu_max_xy, second_zifu_center_y});
            } else if (j == 2) {
                third_zifu_min_xy = cv::Point2f(min_x, min_y);
                third_zifu_max_xy = cv::Point2f(max_x, max_y);
                third_zifu_center_y = (max_y + min_y) / 2.0f;
                zifu_center_list.push_back({third_zifu_min_xy, third_zifu_max_xy, third_zifu_center_y});
            }

            char_images.push_back(roi.clone());
            char_centers.push_back(cv::Point2i((max_x + min_x) / 2, (max_y + min_y) / 2));
        }

        // Direction determination and luhao relocation (only for 3 chars)
        if (angle_params.size() >= 1 && angle_params.size() < 4) {
            int img_zhengfan = 3;
            int luhao_dingwei = 3;

            if (angle_params.size() == 3) {
                if (isFirstNumLargest(guige_max_xy.y, second_zifu_max_xy.y, third_zifu_max_xy.y)) {
                    img_zhengfan = 0;
                } else if (isFirstNumSmallest(guige_min_xy.y, second_zifu_min_xy.y, third_zifu_min_xy.y)) {
                    img_zhengfan = 180;
                }

                if (img_zhengfan == 0) {
                    std::sort(zifu_center_list.begin(), zifu_center_list.end(),
                        [](const zifu_center_info& a, const zifu_center_info& b) {
                            return a.center_y > b.center_y;
                        });
                    if (zifu_center_list.size() == 3) {
                        paihao_max_xy = zifu_center_list[1].pt2;
                        paihao_min_xy = zifu_center_list[1].pt1;
                        paihao_next_zifu_max_xy = zifu_center_list[2].pt2;
                        paihao_next_zifu_min_xy = zifu_center_list[2].pt1;
                    }
                } else if (img_zhengfan == 180) {
                    std::sort(zifu_center_list.begin(), zifu_center_list.end(),
                        [](const zifu_center_info& a, const zifu_center_info& b) {
                            return a.center_y < b.center_y;
                        });
                    if (zifu_center_list.size() == 3) {
                        paihao_max_xy = zifu_center_list[1].pt2;
                        paihao_min_xy = zifu_center_list[1].pt1;
                        paihao_next_zifu_max_xy = zifu_center_list[2].pt2;
                        paihao_next_zifu_min_xy = zifu_center_list[2].pt1;
                    }
                }

                // Check if luhao is already located
                if (img_zhengfan == 0) {
                    int vertical_dis = static_cast<int>(std::abs(paihao_next_zifu_max_xy.y - paihao_min_xy.y));
                    luhao_dingwei = (vertical_dis <= 10) ? 1 : 0;
                } else if (img_zhengfan == 180) {
                    int vertical_dis = static_cast<int>(std::abs(paihao_next_zifu_min_xy.y - paihao_max_xy.y));
                    luhao_dingwei = (vertical_dis <= 10) ? 1 : 0;
                }

                // Relocate luhao if not located
                if (luhao_dingwei == 0) {
                    int first_zifu_width, first_zifu_height;
                    int luhao_min_x, luhao_min_y, luhao_max_x, luhao_max_y;

                    if (img_zhengfan == 0) {
                        first_zifu_width = static_cast<int>(std::abs(paihao_next_zifu_max_xy.x - paihao_next_zifu_min_xy.x));
                        first_zifu_height = static_cast<int>(std::abs(paihao_next_zifu_max_xy.y - paihao_min_xy.y)) - 3;
                        int luhao_width = first_zifu_width + 40;

                        luhao_min_x = static_cast<int>(paihao_next_zifu_min_xy.x);
                        luhao_min_y = static_cast<int>(paihao_next_zifu_max_xy.y);
                        luhao_max_x = luhao_min_x + luhao_width;
                        luhao_max_y = luhao_min_y + first_zifu_height;

                        cv::Rect luhao_rect = safeROI(luhao_min_x, luhao_min_y, luhao_width, first_zifu_height, result_0.cols, result_0.rows);
                        if (luhao_rect.width > 0 && luhao_rect.height > 0) {
                            cv::Mat roi_luhao = result_0(luhao_rect);
                            char_images.push_back(roi_luhao.clone());
                            char_centers.push_back(cv::Point2i(
                                luhao_width / 2 + static_cast<int>(paihao_next_zifu_min_xy.x),
                                first_zifu_height / 2 + static_cast<int>(paihao_next_zifu_max_xy.y)));
                        }
                    } else if (img_zhengfan == 180) {
                        first_zifu_width = static_cast<int>(std::abs(paihao_next_zifu_max_xy.x - paihao_next_zifu_min_xy.x));
                        first_zifu_height = static_cast<int>(std::abs(paihao_next_zifu_min_xy.y - paihao_max_xy.y)) - 3;
                        int luhao_width = first_zifu_width + 40;
                        int luhao_left_top_x = static_cast<int>(paihao_next_zifu_min_xy.x) - 40;

                        luhao_min_x = luhao_left_top_x;
                        luhao_min_y = static_cast<int>(paihao_max_xy.y);
                        luhao_max_x = luhao_left_top_x + luhao_width;
                        luhao_max_y = luhao_min_y + first_zifu_height;

                        cv::Rect luhao_rect = safeROI(luhao_min_x, luhao_min_y, luhao_width, first_zifu_height, result_0.cols, result_0.rows);
                        if (luhao_rect.width > 0 && luhao_rect.height > 0) {
                            cv::Mat roi_luhao = result_0(luhao_rect);
                            char_images.push_back(roi_luhao.clone());
                            char_centers.push_back(cv::Point2i(
                                luhao_width / 2 + luhao_left_top_x,
                                first_zifu_height / 2 + static_cast<int>(paihao_max_xy.y)));
                        }
                    }

                    // Swap to put luhao before the last char
                    if (char_images.size() == 4) {
                        std::swap(char_images[char_images.size() - 1], char_images[char_images.size() - 2]);
                        std::swap(char_centers[char_centers.size() - 1], char_centers[char_centers.size() - 2]);
                    }
                }
            }
        }

        // OCR recognition
        auto recognized_chars = recognizeChars(char_images);
        for (size_t i = 0; i < recognized_chars.size(); i++) {
            if (i < char_centers.size()) {
                recognized_chars[i].center_x = char_centers[i].x;
                recognized_chars[i].center_y = char_centers[i].y;
            }
        }

        if (recognized_chars.empty()) {
            std::cerr << "[PenmaRec] OCR recognition failed for label " << kk << std::endl;
            continue;
        }

        // Direction determination from OCR results
        int direction_flag = 0;
        determineOrientation(recognized_chars, angle_params, direction_flag);

        // Rotate final image if needed
        cv::Mat final_rotated;
        if (direction_flag == 180) {
            cv::flip(result_0, final_rotated, -1);
        } else {
            final_rotated = result_0.clone();
        }
        result.rotated_image = final_rotated;

        // Post-process: find luhao, correct characters
        std::string ocr_str = postprocessResult(recognized_chars, heat_str, label.class_id);
        if (!ocr_str.empty()) {
            ocr_parts.push_back(ocr_str);
            any_success = true;
        }

        result.characters.insert(result.characters.end(), recognized_chars.begin(), recognized_chars.end());
    }

    if (any_success) {
        result.success = true;
        result.ocr_result.clear();
        for (size_t i = 0; i < ocr_parts.size(); i++) {
            if (i < result.labels.size()) {
                std::string label_type = (result.labels[i].class_id == 0) ? "circle" : "hexagon";
                result.ocr_result += label_type + ocr_parts[i];
                if (i < ocr_parts.size() - 1) {
                    result.ocr_result += ",";
                }
            }
        }
    }

    return result;
}

PenmaRecResult PenmaRecInference::recognize(const std::string& image_path, const std::string& heat_str)
{
    cv::Mat image = cv::imread(image_path);
    return recognize(image, heat_str);
}

std::vector<PenmaLabelInfo> PenmaRecInference::detectLabels(const cv::Mat& image)
{
    std::vector<PenmaLabelInfo> result;
    auto detections = m_label_detector->detect(image);

    for (const auto& det : detections) {
        cv::Rect safe_rect = safeROI(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, image.cols, image.rows);
        if (safe_rect.width <= 0 || safe_rect.height <= 0) continue;

        cv::Mat roi = image(safe_rect);
        if (roi.empty()) continue;

        PenmaLabelInfo info;
        info.class_id = det.classId;
        info.bbox = safe_rect;
        info.roi_image = roi.clone();
        result.push_back(info);
    }

    std::cout << "[PenmaRec] Detected " << result.size() << " labels" << std::endl;
    return result;
}

std::vector<OcrAngleParams> PenmaRecInference::detectChars(const cv::Mat& label_roi, cv::Mat& rotated_image)
{
    std::vector<OcrAngleParams> jiaozheng_result;

    auto detections = m_zifu_detector->detect(label_roi);
    if (detections.empty()) {
        std::cerr << "[PenmaRec] No chars detected" << std::endl;
        return jiaozheng_result;
    }

    for (const auto& det : detections) {
        cv::Rect safe_rect = safeROI(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, label_roi.cols, label_roi.rows);
        if (safe_rect.width <= 0 || safe_rect.height <= 0) continue;

        // Create mask from contour for Minrect
        cv::Mat mask_mat = createMaskFromContour(safe_rect, det.mask);
        if (mask_mat.empty()) {
            // Fallback: use bbox as mask
            mask_mat = cv::Mat::ones(safe_rect.height, safe_rect.width, CV_8UC1) * 255;
        }

        OcrAngleParams angle_result = Minrect(mask_mat);

        OcrAngleParams oap;
        oap.angle = angle_result.angle;
        oap.ocrCenterX = angle_result.ocrCenterX;
        oap.ocrCenterY = angle_result.ocrCenterY;
        oap.class_id = det.classId;
        oap.p0 = angle_result.p0;
        oap.p1 = angle_result.p1;
        oap.p2 = angle_result.p2;
        oap.p3 = angle_result.p3;
        oap.left_top.x = safe_rect.x;
        oap.left_top.y = safe_rect.y;
        oap.height = angle_result.height;

        jiaozheng_result.push_back(oap);
    }

    std::cout << "[PenmaRec] Detected " << jiaozheng_result.size() << " chars" << std::endl;
    return jiaozheng_result;
}

std::vector<PenmaCharInfo> PenmaRecInference::recognizeChars(const std::vector<cv::Mat>& char_images)
{
    std::vector<PenmaCharInfo> result;

    for (const auto& img : char_images) {
        if (img.empty()) continue;

        PenmaCharInfo info;
        info.char_image = img.clone();

        try {
            auto ocr_result = m_ocr->recognize_only(img);
            info.text = ocr_result.text;

            if (m_cls) {
                auto cls_result = m_cls->classify(img);
                info.cls_label = cls_result.classId;
            } else {
                info.cls_label = 0;
            }
        } catch (const std::exception& e) {
            std::cerr << "[PenmaRec] OCR failed: " << e.what() << std::endl;
            info.text = "";
            info.cls_label = -1;
        }

        result.push_back(info);
    }

    return result;
}

void PenmaRecInference::determineOrientation(std::vector<PenmaCharInfo>& chars,
                                              const std::vector<OcrAngleParams>& angle_params,
                                              int& direction_flag)
{
    int num_0 = 0, num_180 = 0;
    for (const auto& ch : chars) {
        if (ch.cls_label == 0) num_0++;
        else if (ch.cls_label == 1) num_180++;
    }

    if (num_0 >= num_180) {
        direction_flag = 0;
        std::sort(chars.begin(), chars.end(),
            [](const PenmaCharInfo& a, const PenmaCharInfo& b) {
                return a.center_y < b.center_y;
            });
    } else {
        direction_flag = 180;
        std::sort(chars.begin(), chars.end(),
            [](const PenmaCharInfo& a, const PenmaCharInfo& b) {
                return a.center_y > b.center_y;
            });
    }
}

std::string PenmaRecInference::postprocessResult(std::vector<PenmaCharInfo>& chars,
                                                   const std::string& heat_str,
                                                   int label_class_id)
{
    std::string ocr_str;

    // Find luhao by matching with heat_str
    if (!heat_str.empty() && !chars.empty()) {
        std::vector<int> match_counts;
        for (const auto& ch : chars) {
            match_counts.push_back(countCommonChars(ch.text, heat_str));
        }

        auto max_it = std::max_element(match_counts.begin(), match_counts.end());
        if (max_it != match_counts.end()) {
            size_t max_index = std::distance(match_counts.begin(), max_it);
            int max_value = match_counts[max_index];

            if (max_value >= 4) {
                std::string zifu = chars[max_index].text;
                if (!zifu.empty()) {
                    zifu[0] = '2';
                    std::replace(zifu.begin(), zifu.end(), 'R', '5');
                    std::replace(zifu.begin(), zifu.end(), 'S', '5');
                    if (zifu.size() > 1) {
                        std::replace(zifu.begin(), zifu.end() - 1, 'A', '4');
                    }
                    chars[max_index].text = zifu;
                }
                ocr_str = "#" + chars[max_index].text;
                return ocr_str;
            }
        }
    }

    // Fallback: find luhao by heuristics
    for (auto& ch : chars) {
        if (ch.text.size() <= 1) continue;

        if (ch.text.size() > 7) {
            if (!ch.text.empty() && '0' <= ch.text[0] && ch.text[0] <= '9') {
                std::string zifu = ch.text;
                if (!zifu.empty()) {
                    zifu[0] = '2';
                    std::replace(zifu.begin(), zifu.end(), 'R', '5');
                    std::replace(zifu.begin(), zifu.end(), 'S', '5');
                    if (zifu.size() > 1) {
                        std::replace(zifu.begin(), zifu.end() - 1, 'A', '4');
                    }
                    ch.text = zifu;
                }
                ocr_str = "#" + ch.text;
                break;
            }
        }
    }

    return ocr_str;
}

cv::Mat PenmaRecInference::createMaskFromContour(const cv::Rect& bbox, const std::vector<cv::Point>& contour)
{
    if (contour.empty()) return {};

    cv::Mat mask(bbox.height, bbox.width, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> local_contour;
    for (const auto& pt : contour) {
        int lx = pt.x - bbox.x;
        int ly = pt.y - bbox.y;
        lx = std::max(0, std::min(lx, bbox.width - 1));
        ly = std::max(0, std::min(ly, bbox.height - 1));
        local_contour.push_back(cv::Point(lx, ly));
    }

    if (local_contour.size() >= 3) {
        std::vector<std::vector<cv::Point>> contours = {local_contour};
        cv::fillPoly(mask, contours, cv::Scalar(255));
    }

    return mask;
}

cv::Rect PenmaRecInference::safeROI(int x, int y, int w, int h, int img_w, int img_h)
{
    int max_x = x + w;
    int max_y = y + h;

    x = std::max(0, std::min(x, img_w - 1));
    y = std::max(0, std::min(y, img_h - 1));
    max_x = std::max(0, std::min(max_x, img_w - 1));
    max_y = std::max(0, std::min(max_y, img_h - 1));

    return cv::Rect(x, y, max_x - x, max_y - y);
}

int countCommonChars(const std::string& str1, const std::string& str2)
{
    int commonCount = 0;
    int littlenumber = static_cast<int>(std::min(str1.size(), str2.size()));
    for (int i = 0; i < littlenumber; i++) {
        if (str1[i] == str2[i]) {
            commonCount++;
        }
    }
    return commonCount;
}

bool isFirstNumLargest(float a, float b, float c)
{
    return a >= b && a >= c;
}

bool isFirstNumSmallest(float a, float b, float c)
{
    return a <= b && a <= c;
}
