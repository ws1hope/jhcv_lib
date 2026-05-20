#include <stdio.h>
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
    std::cout << "Local test mode:" << std::endl;
    std::cout << "  " << progName << " --test -c config.yaml -i image.jpg -H heat -s station" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --config   Path to config YAML file (default: config.yaml)" << std::endl;
    std::cout << "  --test         Run in local test mode (single image inference)" << std::endl;
    std::cout << "  -i, --image    Image path for local test (use # to separate multiple)" << std::endl;
    std::cout << "  -H, --heat     Heat number for local test" << std::endl;
    std::cout << "  -s, --station  Station ID for local test" << std::endl;
    std::cout << "  -v, --verbose  Verbose output for local test" << std::endl;
    std::cout << "  -h, --help     Show this help" << std::endl;
}

int main(int argc, char* argv[])
{
    std::string config_path = "config.yaml";
    bool test_mode = false;
    std::string test_image;
    std::string test_heat = "";
    int test_station = 0;

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
        } else if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        }
    }

    if (test_mode && test_image.empty()) {
        std::cerr << "[ERROR] --test mode requires -i <image_path>" << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    if (!test_mode) {
        std::cout << "=== dabang_jiguang OCR HTTP service ===" << std::endl;
    }
    std::cout << "[INFO] Config: " << config_path << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::unique_ptr<JHDeepCore::OCRService> service_ptr;
    try {
        service_ptr = std::make_unique<JHDeepCore::OCRService>(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize models: " << e.what() << std::endl;
        return 1;
    }
    JHDeepCore::OCRService& service = *service_ptr;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] Model init time: " << duration.count() << " ms" << std::endl;
    if (!test_mode) {
        std::cout << "[INFO] Server: " << service.config().host << ":" << service.config().port << std::endl;
    }
    std::cout << "[INFO] Device: " << service.config().device << std::endl;

    if (test_mode) {
        return service.runLocalTest(test_image, test_heat, test_station);
    }

    httplib::Server server;

    server.Get("/hello", [](const httplib::Request&, httplib::Response& resp) {
        resp.set_content("Hello World! dabang_jiguang OCR service running.", "text/plain");
        resp.status = 200;
    });

    server.Post("/character_recognition", [&service](const httplib::Request& req, httplib::Response& res) {
        json result = service.handleRequest(req.body);
        res.set_content(result.dump(), "application/json");
    });

    server.Get("/stop", [&](const httplib::Request&, httplib::Response&) {
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

    std::cout << "[INFO] HTTP server start: " << service.config().host << ":" << service.config().port << std::endl;
    std::cout << "[INFO] POST /character_recognition" << std::endl;
    std::cout << "[INFO] GET  /hello" << std::endl;
    std::cout << "[INFO] GET  /stop" << std::endl;

    if (!server.listen(service.config().host, service.config().port)) {
        std::cerr << "[ERROR] Failed to start server on "
             << service.config().host << ":" << service.config().port << std::endl;
        return 1;
    }

    std::cout << "[INFO] Server stopped." << std::endl;
    return 0;
}
