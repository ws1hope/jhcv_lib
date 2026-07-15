#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "JHDeepCore.h"
#include "jiuyang_pipeline.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static bool ensureDirectoryExists(const std::string &path)
{
#ifdef _WIN32
    if (_access(path.c_str(), 0) != 0) {
        return _mkdir(path.c_str()) == 0;
    }
    return true;
#else
    if (access(path.c_str(), F_OK) != 0) {
        return mkdir(path.c_str(), 0755) == 0;
    }
    return true;
#endif
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [label_path] [device_id]"
                  << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = (argc > 3) ? argv[3] : "";
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;

    try {
        JHDeepCore::JiuyangPipeline pipeline(model_path, label_path, device_id);

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        cv::Mat result_image = pipeline.process(image);

        const auto &bend_results = pipeline.lastBendResults();
        std::cout << "Billet bend results: " << bend_results.size() << std::endl;
        for (const auto &result : bend_results) {
            std::cout << "  id=" << result.id
                      << ", valid=" << (result.valid ? "true" : "false");
            if (result.valid) {
                std::cout << ", bent=" << (result.bent ? "true" : "false")
                          << ", bend95=" << result.bend95
                          << ", bendRms=" << result.bendRms
                          << ", bendMax=" << result.bendMax
                          << ", medianWidth=" << result.medianWidth
                          << ", validLengthRatio=" << result.validLengthRatio
                          << ", stoppedAtMerge="
                          << (result.stoppedAtMerge ? "true" : "false");
            }
            std::cout << std::endl;
        }

        // 创建 result 文件夹（如果不存在）
        if (!ensureDirectoryExists("result")) {
            std::cerr << "Warning: Failed to create result directory" << std::endl;
        }

        // 保存结果（按输入图片文件名保存到 result/ 目录）
        std::string filename = std::filesystem::path(image_path).filename().string();
        std::string output_path = "result/" + filename;
        if (cv::imwrite(output_path, result_image)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Failed to save result to: " << output_path << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
