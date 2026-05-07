#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace JHDeepCore {
namespace utils {

struct ModelConfig {
    std::string task_type;
    int class_scale;
    cv::Size img_scale;
    std::vector<std::string> class_names;
    std::vector<float> mean;
    std::vector<float> stddev;
    int batch_size;

    float conf_threshold;
    float iou_threshold;

    ModelConfig()
        : task_type("classification"), class_scale(224), img_scale(cv::Size(512, 512)),
          mean({123.675f / 255.0f, 116.28f / 255.0f, 103.53f / 255.0f}),
          stddev({58.395f / 255.0f, 57.12f / 255.0f, 57.375f / 255.0f}), batch_size(1), conf_threshold(0.25f),
          iou_threshold(0.45f) {}
};

class ConfigLoader {
  public:
    static ModelConfig LoadFromYaml(const std::string &yaml_path);
    static std::string GetConfigPath(const std::string &model_path);
    static bool ValidateConfig(const ModelConfig &config);
};

} // namespace utils
} // namespace JHDeepCore
