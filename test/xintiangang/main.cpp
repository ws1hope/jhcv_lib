#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>

#include "httplib.h"
#include "json.hpp"
#include "JHDeepCore.h"
#include "file_utils.h"

using json = nlohmann::json;

static void printHelp(const char* progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Server mode (default):" << std::endl;
    std::cout << "  " << progName << " -c config.yaml" << std::endl;
    std::cout << std::endl;
    std::cout << "Local test mode (single image or whole folder):" << std::endl;
    std::cout << "  " << progName << " --test -c config.yaml -i image.jpg  -H heat -s station" << std::endl;
    std::cout << "  " << progName << " --test -c config.yaml -i image_dir -H heat -s station" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --config   Path to config YAML file (default: config.yaml)" << std::endl;
    std::cout << "  --test         Run in local test mode" << std::endl;
    std::cout << "  -i, --image    Image file or directory for local test" << std::endl;
    std::cout << "  -H, --heat     Heat number for local test" << std::endl;
    std::cout << "  -s, --station  Station ID for local test" << std::endl;
    std::cout << "  -o, --outdir   Result output directory (default: xintiangang_result)" << std::endl;
    std::cout << "  -h, --help     Show this help" << std::endl;
}

static bool isImageFile(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const char* kImageExts[] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };
    for (const char* image_ext : kImageExts) {
        if (ext == image_ext) {
            return true;
        }
    }
    return false;
}

// -i 允许传单个文件或目录；目录时收集其中的图片并排序
static std::vector<std::string> collectImagePaths(const std::string& input_path)
{
    std::vector<std::string> paths;
    const std::filesystem::path input(input_path);

    if (std::filesystem::is_directory(input)) {
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        std::sort(paths.begin(), paths.end());
    } else if (std::filesystem::is_regular_file(input)) {
        paths.push_back(input_path);
    }

    return paths;
}

int main(int argc, char* argv[])
{
    std::string config_path = "config.yaml";
    bool test_mode = false;
    std::string test_image;
    std::string test_heat = "";
    int test_station = 0;
    std::string test_outdir = "";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--test") {
            test_mode = true;
        } else if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            test_image = argv[++i];
        } else if ((arg == "-H" || arg == "--heat") && i + 1 < argc) {
            test_heat = argv[++i];
        } else if ((arg == "-s" || arg == "--station") && i + 1 < argc) {
            test_station = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--outdir") && i + 1 < argc) {
            test_outdir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        }
    }

    if (test_mode && test_image.empty()) {
        std::cerr << "[ERROR] --test mode requires -i <image_path_or_dir>" << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    if (!test_mode) {
        std::cout << "=== Xintiangang HTTP Service ===" << std::endl;
    }
    std::cout << "[INFO] Config: " << config_path << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::unique_ptr<JHDeepCore::XintiangangService> service_ptr;
    try {
        service_ptr = std::make_unique<JHDeepCore::XintiangangService>(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize models: " << e.what() << std::endl;
        return 1;
    }
    JHDeepCore::XintiangangService& service = *service_ptr;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] Model init time: " << duration.count() << " ms" << std::endl;
    if (!test_mode) {
        std::cout << "[INFO] Server: " << service.config().host << ":" << service.config().port << std::endl;
    }
    std::cout << "[INFO] Device: " << service.config().device << std::endl;

    if (test_mode) {
        std::vector<std::string> images = collectImagePaths(test_image);
        if (images.empty()) {
            std::cerr << "[ERROR] no image found in: " << test_image << std::endl;
            return 1;
        }

        std::cout << "[INFO] Local test: " << images.size()
                  << " image(s) from " << test_image
                  << ", out_dir="
                  << (test_outdir.empty() ? "xintiangang_result" : test_outdir)
                  << std::endl;

        int ok_count = 0;
        int failed_count = 0;
        for (size_t i = 0; i < images.size(); i++) {
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "[" << (i + 1) << "/" << images.size() << "] "
                      << images[i] << std::endl;
            int rc = service.runLocalTest(images[i], test_heat, test_station, test_outdir);
            if (rc == 0) {
                ok_count++;
            } else {
                failed_count++;
            }
        }

        std::cout << "----------------------------------------" << std::endl;
        std::cout << "=== Summary ===" << std::endl;
        std::cout << "  total:  " << images.size() << std::endl;
        std::cout << "  ok:     " << ok_count << std::endl;
        std::cout << "  failed: " << failed_count << std::endl;
        return failed_count == 0 ? 0 : 1;
    }

    httplib::Server server;
    std::string prefix = "/" + service.config().service_name;

    server.Get(prefix + "/hello", [](const httplib::Request&, httplib::Response& resp) {
        resp.set_content("Hello World! Xintiangang service running.", "text/plain");
        resp.status = 200;
    });

    server.Post(prefix,
        [&service](const httplib::Request& req, httplib::Response& res) {
            std::string result = service.handleRequest(req.body);
            res.set_content(result, "application/json");
        });

    server.Get(prefix + "/stop", [&](const httplib::Request&, httplib::Response&) {
        std::cout << "[INFO] Received stop signal." << std::endl;
        server.stop();
    });

    server.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        char buf[BUFSIZ];
        snprintf(buf, sizeof(buf), "<p>Error Status: <span style='color:red;'>%d</span></p>", res.status);
        res.set_content(buf, "text/html");
    });

    server.set_keep_alive_max_count(10);
    server.set_keep_alive_timeout(10);
    server.set_payload_max_length(1024 * 1024 * 512);

    std::cout << "[INFO] HTTP server start: " << service.config().host
              << ":" << service.config().port << std::endl;
    std::cout << "[INFO] POST " << prefix << std::endl;
    std::cout << "[INFO] GET  " << prefix << "/hello" << std::endl;
    std::cout << "[INFO] GET  " << prefix << "/stop" << std::endl;

    if (!server.listen(service.config().host, service.config().port)) {
        std::cerr << "[ERROR] Failed to start server on "
             << service.config().host << ":" << service.config().port << std::endl;
        return 1;
    }

    std::cout << "[INFO] Server stopped." << std::endl;
    return 0;
}
