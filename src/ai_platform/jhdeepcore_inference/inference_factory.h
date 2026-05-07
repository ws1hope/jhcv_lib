#pragma once

#include <memory>
#include <string>
#include <vector>
#include "jhdeepcore_inference/base_inference.h"

namespace JHDeepCore {
namespace inference {

class InferenceFactory {
  public:
    static std::unique_ptr<BaseInference> CreateInference(const std::string &model_path,
                                                          const std::string &device = "cpu",
                                                          const std::vector<std::string> &class_names = {});

    static bool IsSupportedModel(const std::string &model_path);

  private:
    static std::string GetModelType(const std::string &model_path);
};

} // namespace inference
} // namespace JHDeepCore
