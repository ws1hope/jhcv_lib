#include "jhdeepcore_utils/config_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef YAMLCPP_FOUND
#include <yaml-cpp/yaml.h>
#endif

namespace JHDeepCore {
namespace utils {

std::string ConfigLoader::GetConfigPath(const std::string &model_path) {
    std::string config_path = model_path;
    size_t pos = config_path.rfind(".onnx");
    if (pos != std::string::npos) {
        config_path.replace(pos, 5, ".yaml");
    } else {
        config_path += ".yaml";
    }
    return config_path;
}

ModelConfig ConfigLoader::LoadFromYaml(const std::string &yaml_path) {
    ModelConfig config;

#ifdef YAMLCPP_FOUND
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_path);

        if (yaml_config["task_type"]) {
            config.task_type = yaml_config["task_type"].as<std::string>();
        }

        if (yaml_config["class_scale"]) {
            config.class_scale = yaml_config["class_scale"].as<int>();
        }

        if (yaml_config["img_scale"]) {
            auto scale = yaml_config["img_scale"].as<std::vector<int>>();
            if (scale.size() >= 2) {
                config.img_scale = cv::Size(scale[1], scale[0]);
            }
        }

        if (yaml_config["class_names"]) {
            config.class_names = yaml_config["class_names"].as<std::vector<std::string>>();
        }

        if (yaml_config["mean"]) {
            auto mean_vec = yaml_config["mean"].as<std::vector<float>>();
            config.mean.clear();
            for (float m : mean_vec) {
                config.mean.push_back(m / 255.0f);
            }
        }

        if (yaml_config["std"]) {
            auto std_vec = yaml_config["std"].as<std::vector<float>>();
            config.stddev.clear();
            for (float s : std_vec) {
                config.stddev.push_back(s / 255.0f);
            }
        }

        if (yaml_config["batch_size"]) {
            config.batch_size = yaml_config["batch_size"].as<int>();
        }

        if (yaml_config["conf_threshold"]) {
            config.conf_threshold = yaml_config["conf_threshold"].as<float>();
        }

        if (yaml_config["iou_threshold"]) {
            config.iou_threshold = yaml_config["iou_threshold"].as<float>();
        }

    } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to load YAML config: " << e.what() << std::endl;
        std::cerr << "Using default config" << std::endl;
    }
#else
    std::cerr << "Warning: YAML-CPP not found, using default config" << std::endl;
#endif

    return config;
}

bool ConfigLoader::ValidateConfig(const ModelConfig &config) {
    std::vector<std::string> valid_task_types = {"classification", "segmentation", "detection", "instance"};
    if (std::find(valid_task_types.begin(), valid_task_types.end(), config.task_type) == valid_task_types.end()) {
        std::cerr << "Error: Invalid task type: " << config.task_type << std::endl;
        return false;
    }

    if (config.class_names.empty()) {
        std::cerr << "Warning: Class names list is empty" << std::endl;
    }

    if (config.mean.size() != 3 || config.stddev.size() != 3) {
        std::cerr << "Error: Normalization params count incorrect (expected 3 channels)" << std::endl;
        return false;
    }

    return true;
}

} // namespace utils
} // namespace JHDeepCore
