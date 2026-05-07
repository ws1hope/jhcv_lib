#include "jhdeepcore_inference/base_inference.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace JHDeepCore {
namespace inference {

BaseInference::BaseInference(const std::string &model_path, const std::string &device,
                             const std::vector<std::string> &class_names)
    : model_path_(model_path), device_(device), class_names_(class_names), conf_threshold_(0.25f),
      iou_threshold_(0.45f), letterbox_pad_(0, 0), letterbox_gain_(1.0f) {

    std::string config_path = utils::ConfigLoader::GetConfigPath(model_path);
    config_ = utils::ConfigLoader::LoadFromYaml(config_path);

    if (class_names_.empty() && !config_.class_names.empty()) {
        class_names_ = config_.class_names;
    }

    if (config_.task_type == "detection") {
        if (config_.conf_threshold > 0) {
            conf_threshold_ = config_.conf_threshold;
        }
        if (config_.iou_threshold > 0) {
            iou_threshold_ = config_.iou_threshold;
        }
    }
}

cv::Mat BaseInference::PreprocessImageCommon(const cv::Mat &image) {
    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else {
        rgb_image = image.clone();
    }

    original_image_size_ = rgb_image.size();

    cv::Size resize_size;
    if (config_.task_type == "segmentation" && config_.img_scale.width > 0) {
        resize_size = config_.img_scale;
    } else {
        resize_size = cv::Size(config_.class_scale, config_.class_scale);
    }

    cv::Mat resized;
    cv::resize(rgb_image, resized, resize_size);

    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    for (size_t i = 0; i < channels.size() && i < config_.mean.size(); ++i) {
        channels[i] = (channels[i] - config_.mean[i]) / config_.stddev[i];
    }
    cv::merge(channels, normalized);

    return normalized;
}

cv::Mat BaseInference::PreprocessImageDetection(const cv::Mat &image) {
    original_image_size_ = image.size();

    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else {
        rgb_image = image.clone();
    }

    cv::Size target_size = config_.img_scale.width > 0 ? config_.img_scale : cv::Size(640, 640);
    cv::Mat letterboxed = Letterbox(rgb_image, target_size, letterbox_pad_, letterbox_gain_);

    cv::Mat normalized;
    letterboxed.convertTo(normalized, CV_32F, 1.0 / 255.0);

    if (!config_.mean.empty() && !config_.stddev.empty()) {
        std::vector<cv::Mat> channels;
        cv::split(normalized, channels);
        for (size_t i = 0; i < channels.size() && i < config_.mean.size(); ++i) {
            channels[i] = (channels[i] - config_.mean[i]) / config_.stddev[i];
        }
        cv::merge(channels, normalized);
    }

    return normalized;
}

ClassificationResult BaseInference::ProcessClassificationOutput(const std::vector<float> &output) {
    ClassificationResult result;

    std::vector<float> probabilities = Softmax(output);

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    result.class_id = std::distance(probabilities.begin(), max_it);
    result.confidence = *max_it;

    result.class_name = GetClassName(result.class_id);

    result.probabilities = probabilities;

    return result;
}

SegmentationResult BaseInference::ProcessSegmentationOutput(const std::vector<float> &output,
                                                            const std::vector<int64_t> &output_shape) {
    SegmentationResult result;

    int num_dims = static_cast<int>(output_shape.size());
    int batch_dim = 1;
    int channels = 0;
    int height = 0;
    int width = 0;

    std::string shape_str = "[";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i > 0)
            shape_str += ", ";
        shape_str += std::to_string(output_shape[i]);
    }
    shape_str += "]";

    if (num_dims == 4) {
        batch_dim = static_cast<int>(output_shape[0]);
        channels = static_cast<int>(output_shape[1]);
        height = static_cast<int>(output_shape[2]);
        width = static_cast<int>(output_shape[3]);

        if (batch_dim == -1) {
            batch_dim = 1;
        }

        bool has_dynamic_dim = (channels == -1 || height == -1 || width == -1);
        if (has_dynamic_dim) {
            size_t total_elements = output.size();

            bool channels_known = (channels > 0);
            bool height_known = (height > 0);
            bool width_known = (width > 0);

            if (channels_known && height_known && !width_known) {
                if (batch_dim > 0 && channels > 0 && height > 0) {
                    size_t divisor =
                        static_cast<size_t>(batch_dim) * static_cast<size_t>(channels) * static_cast<size_t>(height);
                    if (divisor == 0) {
                        throw std::runtime_error("Divisor is 0, cannot infer width");
                    }
                    width = static_cast<int>(total_elements / divisor);
                    if (width <= 0) {
                        throw std::runtime_error("Inferred width invalid: " + std::to_string(width));
                    }
                }
            } else if (channels_known && !height_known && width_known) {
                if (batch_dim > 0 && channels > 0 && width > 0) {
                    size_t divisor =
                        static_cast<size_t>(batch_dim) * static_cast<size_t>(channels) * static_cast<size_t>(width);
                    if (divisor == 0) {
                        throw std::runtime_error("Divisor is 0, cannot infer height");
                    }
                    height = static_cast<int>(total_elements / divisor);
                }
            } else if (!channels_known && height_known && width_known) {
                if (batch_dim > 0 && height > 0 && width > 0) {
                    size_t divisor =
                        static_cast<size_t>(batch_dim) * static_cast<size_t>(height) * static_cast<size_t>(width);
                    if (divisor == 0) {
                        throw std::runtime_error("Divisor is 0, cannot infer channels");
                    }
                    channels = static_cast<int>(total_elements / divisor);
                }
            } else {
                throw std::runtime_error("Cannot infer output shape, too many dynamic dims: " + shape_str);
            }
        }
    } else if (num_dims == 3) {
        channels = static_cast<int>(output_shape[0]);
        height = static_cast<int>(output_shape[1]);
        width = static_cast<int>(output_shape[2]);

        bool has_dynamic_dim = (channels == -1 || height == -1 || width == -1);
        if (has_dynamic_dim) {
            size_t total_elements = output.size();

            bool channels_known = (channels > 0);
            bool height_known = (height > 0);
            bool width_known = (width > 0);

            if (channels_known && height_known && !width_known) {
                if (channels > 0 && height > 0) {
                    size_t divisor = static_cast<size_t>(channels) * static_cast<size_t>(height);
                    width = static_cast<int>(total_elements / divisor);
                }
            } else if (channels_known && !height_known && width_known) {
                if (channels > 0 && width > 0) {
                    size_t divisor = static_cast<size_t>(channels) * static_cast<size_t>(width);
                    height = static_cast<int>(total_elements / divisor);
                }
            } else if (!channels_known && height_known && width_known) {
                if (height > 0 && width > 0) {
                    size_t divisor = static_cast<size_t>(height) * static_cast<size_t>(width);
                    channels = static_cast<int>(total_elements / divisor);
                }
            } else {
                throw std::runtime_error("Cannot infer output shape, too many dynamic dims: " + shape_str);
            }
        }
    } else {
        throw std::runtime_error("Unsupported output shape dimensions: " + std::to_string(num_dims));
    }

    if (channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error("Invalid output shape: channels=" + std::to_string(channels) +
                                 ", height=" + std::to_string(height) + ", width=" + std::to_string(width));
    }

    size_t expected_size = static_cast<size_t>(batch_dim) * static_cast<size_t>(channels) *
                           static_cast<size_t>(height) * static_cast<size_t>(width);
    if (output.size() != expected_size) {
        throw std::runtime_error("Output data size mismatch: expected " + std::to_string(expected_size) +
                                 ", actual " + std::to_string(output.size()));
    }

    int data_offset = 0;
    if (num_dims == 4 && batch_dim > 1) {
        data_offset = 0;
    }

    if (height <= 0 || width <= 0) {
        throw std::runtime_error("Invalid mask dimensions: height=" + std::to_string(height) +
                                 ", width=" + std::to_string(width));
    }

    cv::Mat segmentation_mask(height, width, CV_8UC1);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            std::vector<float> pixel_logits(channels);
            for (int c = 0; c < channels; ++c) {
                int idx = data_offset + c * height * width + h * width + w;
                if (idx >= static_cast<int>(output.size())) {
                    throw std::runtime_error("Output data index out of bounds: " + std::to_string(idx));
                }
                pixel_logits[c] = output[idx];
            }

            std::vector<float> probabilities = Softmax(pixel_logits);

            auto max_it = std::max_element(probabilities.begin(), probabilities.end());
            int class_id = static_cast<int>(std::distance(probabilities.begin(), max_it));

            segmentation_mask.at<uchar>(h, w) = static_cast<uchar>(class_id);
        }
    }

    if (segmentation_mask.size() != original_image_size_) {
        segmentation_mask = ResizeMaskToOriginal(segmentation_mask, original_image_size_);
    }

    result.segmentation_mask = segmentation_mask;
    result.num_classes = channels;
    result.image_shape = original_image_size_;
    result.class_names = class_names_;

    return result;
}

std::string BaseInference::GetClassName(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
        return class_names_[class_id];
    }
    return "class_" + std::to_string(class_id);
}

std::vector<float> BaseInference::Softmax(const std::vector<float> &logits) {
    if (logits.empty()) {
        return {};
    }

    float max_val = *std::max_element(logits.begin(), logits.end());

    std::vector<float> exp_values(logits.size());
    std::transform(logits.begin(), logits.end(), exp_values.begin(),
                   [max_val](float x) { return std::exp(x - max_val); });

    float sum = std::accumulate(exp_values.begin(), exp_values.end(), 0.0f);

    std::vector<float> probabilities(exp_values.size());
    std::transform(exp_values.begin(), exp_values.end(), probabilities.begin(), [sum](float x) { return x / sum; });

    return probabilities;
}

cv::Mat BaseInference::Letterbox(const cv::Mat &img, const cv::Size &new_shape, cv::Point2i &pad, float &gain) {
    float scale_w = static_cast<float>(new_shape.width) / img.cols;
    float scale_h = static_cast<float>(new_shape.height) / img.rows;
    gain = std::min(scale_w, scale_h);

    int new_width = static_cast<int>(std::round(img.cols * gain));
    int new_height = static_cast<int>(std::round(img.rows * gain));

    cv::Mat resized;
    if (img.size() != cv::Size(new_width, new_height)) {
        cv::resize(img, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img.clone();
    }

    float dw = (new_shape.width - new_width) / 2.0f;
    float dh = (new_shape.height - new_height) / 2.0f;

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    pad = cv::Point2i(left, top);

    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return letterboxed;
}

cv::Mat BaseInference::ResizeMaskToOriginal(const cv::Mat &mask, const cv::Size &original_size) {
    cv::Mat resized;
    cv::resize(mask, resized, original_size, 0, 0, cv::INTER_NEAREST);
    return resized;
}

DetectionResult BaseInference::ProcessDetectionOutput(const std::vector<float> &output,
                                                      const std::vector<int64_t> &output_shape) {
    DetectionResult result;
    result.image_shape = original_image_size_;
    result.num_detections = 0;

    int num_detections = 0;
    int num_info = 0;

    if (output_shape.size() == 3) {
        num_info = static_cast<int>(output_shape[1]);
        num_detections = static_cast<int>(output_shape[2]);
    } else if (output_shape.size() == 2) {
        num_info = static_cast<int>(output_shape[0]);
        num_detections = static_cast<int>(output_shape[1]);
    } else {
        throw std::runtime_error("Unsupported detection output shape dimensions");
    }

    int num_classes = num_info - 4;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; ++i) {
        float x_center = output[0 * num_detections + i];
        float y_center = output[1 * num_detections + i];
        float box_width = output[2 * num_detections + i];
        float box_height = output[3 * num_detections + i];

        float max_class_score = 0.0f;
        int max_class_id = 0;

        for (int c = 0; c < num_classes; ++c) {
            float class_score = output[(4 + c) * num_detections + i];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = c;
            }
        }

        if (max_class_score < conf_threshold_) {
            continue;
        }

        float x1 = (x_center - box_width / 2.0f - letterbox_pad_.x) / letterbox_gain_;
        float y1 = (y_center - box_height / 2.0f - letterbox_pad_.y) / letterbox_gain_;
        float x2 = (x_center + box_width / 2.0f - letterbox_pad_.x) / letterbox_gain_;
        float y2 = (y_center + box_height / 2.0f - letterbox_pad_.y) / letterbox_gain_;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_image_size_.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_image_size_.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_image_size_.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_image_size_.height)));

        cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

        if (bbox.width > 0 && bbox.height > 0) {
            boxes.push_back(bbox);
            confidences.push_back(max_class_score);
            class_ids.push_back(max_class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);

    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        det.class_name = GetClassName(class_ids[idx]);
        result.detections.push_back(det);
    }

    result.num_detections = static_cast<int>(result.detections.size());
    return result;
}

InstanceSegmentationResult BaseInference::ProcessInstanceSegmentationOutput(
    const std::vector<float> &detection_output, const std::vector<int64_t> &detection_output_shape,
    const std::vector<float> &protos_output, const std::vector<int64_t> &protos_output_shape) {
    InstanceSegmentationResult result;
    result.image_shape = original_image_size_;
    result.num_detections = 0;

    int mask_dim = 0;
    int mask_h = 0;
    int mask_w = 0;

    if (protos_output_shape.size() == 4) {
        mask_dim = static_cast<int>(protos_output_shape[1]);
        mask_h = static_cast<int>(protos_output_shape[2]);
        mask_w = static_cast<int>(protos_output_shape[3]);
    } else if (protos_output_shape.size() == 3) {
        mask_dim = static_cast<int>(protos_output_shape[0]);
        mask_h = static_cast<int>(protos_output_shape[1]);
        mask_w = static_cast<int>(protos_output_shape[2]);
    } else {
        throw std::runtime_error("Unsupported instance segmentation proto mask output shape dimensions");
    }

    int num_detections = 0;
    int num_info = 0;
    bool need_transpose = false;

    if (detection_output_shape.size() == 3) {
        int dim1 = static_cast<int>(detection_output_shape[1]);
        int dim2 = static_cast<int>(detection_output_shape[2]);

        if (dim1 < 1000 && dim2 > 1000) {
            num_info = dim1;
            num_detections = dim2;
            need_transpose = true;
        } else if (dim1 > 1000 && dim2 < 1000) {
            num_detections = dim1;
            num_info = dim2;
            need_transpose = false;
        } else {
            if (dim1 >= (4 + mask_dim) && (dim1 - 4 - mask_dim) < 10000) {
                num_info = dim1;
                num_detections = dim2;
                need_transpose = true;
            } else {
                num_detections = dim1;
                num_info = dim2;
                need_transpose = false;
            }
        }
    } else if (detection_output_shape.size() == 2) {
        int dim1 = static_cast<int>(detection_output_shape[0]);
        int dim2 = static_cast<int>(detection_output_shape[1]);

        if (dim1 < 1000 && dim2 > 1000) {
            num_info = dim1;
            num_detections = dim2;
            need_transpose = true;
        } else if (dim1 > 1000 && dim2 < 1000) {
            num_detections = dim1;
            num_info = dim2;
            need_transpose = false;
        } else {
            if (dim1 >= (4 + mask_dim) && (dim1 - 4 - mask_dim) < 10000) {
                num_info = dim1;
                num_detections = dim2;
                need_transpose = true;
            } else {
                num_detections = dim1;
                num_info = dim2;
                need_transpose = false;
            }
        }
    } else {
        throw std::runtime_error("Unsupported instance segmentation detection output shape dimensions");
    }

    int num_classes = num_info - 4 - mask_dim;

    if (num_classes < 0 || num_classes > 10000) {
        if (detection_output_shape.size() == 3) {
            int dim1 = static_cast<int>(detection_output_shape[1]);
            int dim2 = static_cast<int>(detection_output_shape[2]);
            num_info = dim2;
            num_detections = dim1;
            need_transpose = !need_transpose;
            num_classes = num_info - 4 - mask_dim;
        } else if (detection_output_shape.size() == 2) {
            int dim1 = static_cast<int>(detection_output_shape[0]);
            int dim2 = static_cast<int>(detection_output_shape[1]);
            num_info = dim2;
            num_detections = dim1;
            need_transpose = !need_transpose;
            num_classes = num_info - 4 - mask_dim;
        }

        if (num_classes < 0 || num_classes > 10000) {
            throw std::runtime_error("Failed to compute num_classes: num_info=" + std::to_string(num_info) +
                                     ", mask_dim=" + std::to_string(mask_dim));
        }
    }

    std::vector<float> pred_output(num_detections * num_info);
    if (need_transpose) {
        for (int i = 0; i < num_detections; ++i) {
            for (int j = 0; j < num_info; ++j) {
                pred_output[i * num_info + j] = detection_output[j * num_detections + i];
            }
        }
    } else {
        pred_output = detection_output;
    }

    std::vector<float> protos;
    if (protos_output_shape.size() == 4) {
        int protos_size = mask_dim * mask_h * mask_w;
        protos.resize(protos_size);
        std::copy(protos_output.begin(), protos_output.begin() + protos_size, protos.begin());
    } else {
        protos = protos_output;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    for (int i = 0; i < num_detections; ++i) {
        float x_center = pred_output[i * num_info + 0];
        float y_center = pred_output[i * num_info + 1];
        float box_width = pred_output[i * num_info + 2];
        float box_height = pred_output[i * num_info + 3];

        float max_class_score = 0.0f;
        int max_class_id = 0;
        for (int c = 0; c < num_classes; ++c) {
            float class_score = pred_output[i * num_info + 4 + c];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                max_class_id = c;
            }
        }

        if (max_class_score < conf_threshold_) {
            continue;
        }

        std::vector<float> mask_coeff(mask_dim);
        for (int m = 0; m < mask_dim; ++m) {
            mask_coeff[m] = pred_output[i * num_info + 4 + num_classes + m];
        }

        float x1 = (x_center - box_width / 2.0f - letterbox_pad_.x) / letterbox_gain_;
        float y1 = (y_center - box_height / 2.0f - letterbox_pad_.y) / letterbox_gain_;
        float x2 = (x_center + box_width / 2.0f - letterbox_pad_.x) / letterbox_gain_;
        float y2 = (y_center + box_height / 2.0f - letterbox_pad_.y) / letterbox_gain_;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_image_size_.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_image_size_.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_image_size_.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_image_size_.height)));

        cv::Rect bbox(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

        if (bbox.width > 0 && bbox.height > 0) {
            boxes.push_back(bbox);
            confidences.push_back(max_class_score);
            class_ids.push_back(max_class_id);
            mask_coeffs.push_back(mask_coeff);
        }
    }

    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);
    }

    std::vector<cv::Rect> kept_boxes;
    std::vector<std::vector<float>> kept_mask_coeffs;

    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        det.class_name = GetClassName(class_ids[idx]);
        result.detections.push_back(det);
        kept_boxes.push_back(boxes[idx]);
        kept_mask_coeffs.push_back(mask_coeffs[idx]);
    }

    if (!kept_mask_coeffs.empty() && !protos.empty()) {
        result.masks = ProcessInstanceMasks(protos, kept_mask_coeffs, kept_boxes, mask_dim, mask_h, mask_w);
    }

    result.num_detections = static_cast<int>(result.detections.size());
    return result;
}

std::vector<cv::Mat> BaseInference::ProcessInstanceMasks(const std::vector<float> &protos,
                                                         const std::vector<std::vector<float>> &masks_in,
                                                         const std::vector<cv::Rect> &bboxes, int mask_dim, int mask_h,
                                                         int mask_w) {
    std::vector<cv::Mat> masks;

    if (masks_in.empty() || protos.empty()) {
        return masks;
    }

    int num_detections = static_cast<int>(masks_in.size());
    int img_h = original_image_size_.height;
    int img_w = original_image_size_.width;

    int input_height = 640;
    int input_width = 640;
    if (config_.img_scale.width > 0 && config_.img_scale.height > 0) {
        input_height = config_.img_scale.height;
        input_width = config_.img_scale.width;
    }

    for (int i = 0; i < num_detections; ++i) {
        cv::Mat mask_flat(1, mask_h * mask_w, CV_32F);
        for (int j = 0; j < mask_h * mask_w; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < mask_dim; ++k) {
                sum += masks_in[i][k] * protos[k * mask_h * mask_w + j];
            }
            mask_flat.at<float>(0, j) = sum;
        }

        cv::Mat mask(mask_h, mask_w, CV_32F);
        for (int h = 0; h < mask_h; ++h) {
            for (int w = 0; w < mask_w; ++w) {
                mask.at<float>(h, w) = mask_flat.at<float>(0, h * mask_w + w);
            }
        }

        cv::Mat mask_sigmoid;
        cv::exp(-mask, mask_sigmoid);
        mask_sigmoid = 1.0f / (1.0f + mask_sigmoid);

        int top = static_cast<int>(letterbox_pad_.y);
        int left = static_cast<int>(letterbox_pad_.x);
        int bottom = input_height - top;
        int right = input_width - left;

        cv::Mat mask_input;
        cv::resize(mask_sigmoid, mask_input, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

        cv::Mat mask_unpadded = mask_input(cv::Range(top, bottom), cv::Range(left, right));

        cv::Mat mask_orig;
        cv::resize(mask_unpadded, mask_orig, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

        cv::Rect bbox = bboxes[i];
        int x1 = std::max(0, std::min(bbox.x, img_w));
        int y1 = std::max(0, std::min(bbox.y, img_h));
        int x2 = std::max(0, std::min(bbox.x + bbox.width, img_w));
        int y2 = std::max(0, std::min(bbox.y + bbox.height, img_h));

        cv::Mat mask_final = cv::Mat::zeros(img_h, img_w, CV_32F);
        if (x2 > x1 && y2 > y1) {
            cv::Mat mask_cropped = mask_orig(cv::Range(y1, y2), cv::Range(x1, x2));
            mask_cropped.copyTo(mask_final(cv::Range(y1, y2), cv::Range(x1, x2)));
        }

        cv::Mat mask_binary;
        cv::threshold(mask_final, mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
        mask_binary.convertTo(mask_binary, CV_8U, 255.0);

        masks.push_back(mask_binary);
    }

    return masks;
}

} // namespace inference
} // namespace JHDeepCore
