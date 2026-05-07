#include "jhdeepcore_inference/inference_factory.h"
#include "jhdeepcore_inference/onnx_inference.h"
#include <algorithm>
#include <fstream>

namespace JHDeepCore {
namespace inference {

std::string InferenceFactory::GetModelType(const std::string &model_path) {
    size_t dot_pos = model_path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "";
    }

    std::string extension = model_path.substr(dot_pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    return extension;
}

bool InferenceFactory::IsSupportedModel(const std::string &model_path) {
    std::string model_type = GetModelType(model_path);
    return model_type == "onnx";
}

std::unique_ptr<BaseInference> InferenceFactory::CreateInference(const std::string &model_path,
                                                                 const std::string &device,
                                                                 const std::vector<std::string> &class_names) {

    std::ifstream file(model_path);
    if (!file.good()) {
        throw std::runtime_error("Model file does not exist: " + model_path);
    }
    file.close();

    std::string model_type = GetModelType(model_path);

    if (model_type == "onnx") {
        auto inference = std::make_unique<OnnxInference>(model_path, device, class_names);

        if (!inference->LoadModel()) {
            throw std::runtime_error("Failed to load model: " + model_path);
        }

        return inference;
    } else {
        throw std::runtime_error("Unsupported model type: " + model_type);
    }
}

} // namespace inference
} // namespace JHDeepCore
