#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include "httplib.h"
#include "json.hpp"
#include "JHDeepCore.h"
#include "file_utils.h"

using json = nlohmann::json;

static void printHelp(const char *progName)
{
    std::cout << "Usage: " << progName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Server mode (default):" << std::endl;
    std::cout << "  " << progName << " -c config.yaml" << std::endl;
    std::cout << std::endl;
    std::cout << "Local test mode (offline replay, no HTTP):" << std::endl;
    std::cout << "  " << progName
              << " --test -c config.yaml -m metadata.json -d image_dir" << std::endl;
    std::cout << "  (metadata.json: {station_id, pictures_information:[{camera_id, file_key}]};" << std::endl;
    std::cout << "   image_dir contains one image file per file_key)" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --config  Path to config YAML file (default: config.yaml)" << std::endl;
    std::cout << "  --test        Run in local test mode" << std::endl;
    std::cout << "  -m, --meta    metadata JSON file for local test" << std::endl;
    std::cout << "  -d, --dir     Image directory for local test" << std::endl;
    std::cout << "  -h, --help    Show this help" << std::endl;
}

// 读取整个文件到字符串
static bool readFileContent(const std::string &path, std::string &content)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;
    std::ostringstream ss;
    ss << fin.rdbuf();
    content = ss.str();
    return true;
}

// 本地测试：从 metadata 文件与图片目录构造 (metadata, files)
static int runLocalTest(JHDeepCore::TedaiJuanquService &service,
                        const std::string &meta_path, const std::string &image_dir)
{
    std::string metadata;
    if (!readFileContent(meta_path, metadata)) {
        std::cerr << "[ERROR] cannot read metadata file: " << meta_path << std::endl;
        return 1;
    }

    std::map<std::string, std::string> files;
    json root = json::parse(metadata, nullptr, false);
    if (root.is_discarded() || !root.contains("pictures_information")) {
        std::cerr << "[ERROR] invalid metadata json" << std::endl;
        return 1;
    }
    for (const auto &item : root["pictures_information"]) {
        if (!item.is_object() || !item.contains("file_key")) continue;
        std::string file_key = item["file_key"].get<std::string>();
        std::string content;
        std::string path = image_dir + "/" + file_key;
        if (!readFileContent(path, content)) {
            std::cerr << "[WARN] image not found: " << path << std::endl;
            continue;
        }
        files[file_key] = content;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::string result = service.handleRequest(metadata, files);
    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "=== Result ===" << std::endl;
    std::cout << result << std::endl;
    std::cout << "total time: " << total_ms << " ms" << std::endl;
    return 0;
}

int main(int argc, char *argv[])
{
    std::string config_path = "config.yaml";
    bool test_mode = false;
    std::string test_meta;
    std::string test_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--test") {
            test_mode = true;
        } else if ((arg == "-m" || arg == "--meta") && i + 1 < argc) {
            test_meta = argv[++i];
        } else if ((arg == "-d" || arg == "--dir") && i + 1 < argc) {
            test_dir = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        }
    }

    if (test_mode && (test_meta.empty() || test_dir.empty())) {
        std::cerr << "[ERROR] --test mode requires -m <metadata.json> -d <image_dir>"
                  << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    if (!test_mode) {
        std::cout << "=== Tedai Juanqu HTTP Service ===" << std::endl;
    }
    std::cout << "[INFO] Config: " << config_path << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::unique_ptr<JHDeepCore::TedaiJuanquService> service_ptr;
    try {
        service_ptr = std::make_unique<JHDeepCore::TedaiJuanquService>(config_path);
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] Failed to initialize service: " << e.what() << std::endl;
        return 1;
    }
    JHDeepCore::TedaiJuanquService &service = *service_ptr;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] Service init time: " << duration.count() << " ms" << std::endl;
    std::cout << "[INFO] Cameras: " << service.config().cameras.size()
              << " (models lazy-loaded on first request)" << std::endl;

    if (test_mode) {
        return runLocalTest(service, test_meta, test_dir);
    }

    std::cout << "[INFO] Server: " << service.config().host << ":"
              << service.config().port << std::endl;
    std::cout << "[INFO] Device: " << service.config().device << std::endl;

    httplib::Server server;

    server.Get("/hello", [](const httplib::Request &, httplib::Response &resp) {
        resp.set_content("Hello World!", "text/plain");
        resp.status = 200;
    });

    // 旧服务端点：POST /tedai_panjuan_detect，multipart metadata + 图片文件
    server.Post(service.config().url_path,
        [&service](const httplib::Request &req, httplib::Response &res) {
            std::string metadata;
            if (req.has_file("metadata")) {
                metadata = req.get_file_value("metadata").content;
            }
            std::map<std::string, std::string> files;
            for (const auto &kv : req.files) {
                files[kv.first] = kv.second.content;
            }
            std::string result = service.handleRequest(metadata, files);
            // 旧服务响应 Content-Type 为 text/plain，保持兼容
            res.set_content(result, "text/plain");
        });

    server.Get("/stop", [&](const httplib::Request &, httplib::Response &) {
        std::cout << "[INFO] Received stop signal." << std::endl;
        server.stop();
    });

    server.set_error_handler([](const httplib::Request &, httplib::Response &res) {
        char buf[BUFSIZ];
        snprintf(buf, sizeof(buf),
                 "<p>Error Status: <span style='color:red;'>%d</span></p>", res.status);
        res.set_content(buf, "text/html");
    });

    server.set_keep_alive_max_count(10);
    server.set_keep_alive_timeout(10);
    server.set_payload_max_length(1024 * 1024 * 512);

    std::cout << "[INFO] HTTP server start: " << service.config().host << ":"
              << service.config().port << std::endl;
    std::cout << "[INFO] POST " << service.config().url_path << std::endl;
    std::cout << "[INFO] GET  /hello" << std::endl;
    std::cout << "[INFO] GET  /stop" << std::endl;

    if (!server.listen(service.config().host, service.config().port)) {
        std::cerr << "[ERROR] Failed to start server on " << service.config().host
                  << ":" << service.config().port << std::endl;
        return 1;
    }

    std::cout << "[INFO] Server stopped." << std::endl;
    return 0;
}
