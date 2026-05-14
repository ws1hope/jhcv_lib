#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <direct.h>
#include <io.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include "httplib.h"
#include "json.hpp"
#include "JHDeepCore.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

static mutex g_mutex;

struct ServerConfig {
    string host = "0.0.0.0";
    int port = 8080;
    string result_dir = "D:\\CharacterDetect\\result";
    string split_dir = "D:\\CharacterDetect\\result_split";
    string log_dir = "visual_logs";
    string label_detect_model;
    string char_detect_model;
    string ocr_rec_model;
    string ocr_rec_label;
    string device = "cuda";
};

static ServerConfig g_config;
static JHDeepCore::Detector* g_det_label = nullptr;
static JHDeepCore::Detector* g_det_char = nullptr;
static JHDeepCore::OCRRecognizer* g_ocr = nullptr;

static vector<string> splitStringByCsharp(const string& str)
{
    vector<string> tokens;
    istringstream iss(str);
    string token;
    while (getline(iss, token, '#')) {
        tokens.push_back(token);
    }
    return tokens;
}

static bool ensureDirectoryExists(const string& path)
{
    if (_access(path.c_str(), 0) != 0) {
        if (_mkdir(path.c_str()) != 0) {
            string parent = path;
            size_t pos = parent.find_last_of("\\/");
            if (pos != string::npos) {
                ensureDirectoryExists(parent.substr(0, pos));
            }
            _mkdir(path.c_str());
        }
    }
    return true;
}

static void createSplitDirectories(int station_id, const tm* t)
{
    string base = format("%s\\station_%02d\\%d%02d%02d",
        g_config.split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    ensureDirectoryExists(base);

    const char* subdirs[] = {"biaoqian\\ok", "biaoqian\\ng",
                             "gangyin\\ok", "gangyin\\ng",
                             "penma\\ok", "penma\\ng"};
    for (auto s : subdirs) {
        ensureDirectoryExists(base + "\\" + s);
    }
}

static void writeLog(ofstream& fout, const string& msg)
{
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    fout << msg << endl;
    fout << "timestamp[" << sys.wYear << setfill('0') << setw(2) << sys.wMonth
         << setfill('0') << setw(2) << sys.wDay
         << setfill('0') << setw(2) << sys.wHour
         << setfill('0') << setw(2) << sys.wMinute
         << setfill('0') << setw(2) << sys.wSecond << "]" << endl;
}

static Rect safeROI(int x, int y, int w, int h, int img_w, int img_h)
{
    x = max(0, x);
    y = max(0, y);
    w = min(w, img_w - x);
    h = min(h, img_h - y);
    if (w <= 0 || h <= 0) return Rect(0, 0, 0, 0);
    return Rect(x, y, w, h);
}

static ServerConfig loadConfig(const string& config_path)
{
    ServerConfig cfg;
    YAML::Node node = YAML::LoadFile(config_path);

    if (node["server"]) {
        cfg.host = node["server"]["host"].as<string>(cfg.host);
        cfg.port = node["server"]["port"].as<int>(cfg.port);
    }

    if (node["output"]) {
        cfg.result_dir = node["output"]["result_dir"].as<string>(cfg.result_dir);
        cfg.split_dir = node["output"]["split_dir"].as<string>(cfg.split_dir);
        cfg.log_dir = node["output"]["log_dir"].as<string>(cfg.log_dir);
    }

    if (node["models"]) {
        cfg.label_detect_model = node["models"]["label_detect_model"].as<string>("");
        cfg.char_detect_model = node["models"]["char_detect_model"].as<string>("");
        cfg.ocr_rec_model = node["models"]["ocr_rec_model"].as<string>("");
        cfg.ocr_rec_label = node["models"]["ocr_rec_label"].as<string>("");
    }

    if (node["inference"]) {
        cfg.device = node["inference"]["device"].as<string>("cuda");
    }

    return cfg;
}

static string sortCharsByPosition(
    const vector<JHDeepCore::Detection>& char_dets,
    const vector<string>& char_texts)
{
    if (char_dets.empty()) return "";

    vector<pair<int, string>> chars_with_pos;
    // cout <<"检测到" << char_dets.size() << "个字符，准备排序..." << endl;
    for (int i = 0; i < (int)char_dets.size(); i++) {
        chars_with_pos.emplace_back(char_dets[i].bbox.y, char_texts[i]);
    }

    sort(chars_with_pos.begin(), chars_with_pos.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });

    string combined;
    bool first = true;
    for (auto& cp : chars_with_pos) {
        combined += cp.second;
        if (first) {
            combined += "#";
            first = false;
        }
    }
    return combined;
}

static json runRecognitionSingle(
    const Mat& src_img,
    int station_id,
    const string& heat_number,
    int pic_number,
    bool verbose)
{
    json item;

    auto infer_start = chrono::high_resolution_clock::now();

    JHDeepCore::DetectionResult label_result = g_det_label->DetectSingle(src_img);

    if (verbose) {
        cout << "[DEBUG] pic " << pic_number + 1
             << " labels: " << label_result.num_detections << endl;
    }

    string ocr_combined;
    bool success_flag = false;

    if (label_result.num_detections > 0) {
        for (int li = 0; li < label_result.num_detections; li++) {
            auto& label_det = label_result.detections[li];

            if (verbose) {
                cout << "  label[" << li << "] class=" << label_det.class_name
                     << " conf=" << label_det.confidence
                     << " bbox=(" << label_det.bbox.x << "," << label_det.bbox.y
                     << "," << label_det.bbox.width << "," << label_det.bbox.height
                     << ")" << endl;
            }

            Rect label_roi = safeROI(
                label_det.bbox.x, label_det.bbox.y,
                label_det.bbox.width, label_det.bbox.height,
                src_img.cols, src_img.rows);

            if (label_roi.area() <= 0) continue;

            Mat roi_img = src_img(label_roi).clone();

            JHDeepCore::DetectionResult char_result = g_det_char->DetectSingle(roi_img);

            if (verbose) {
                cout << "  chars: " << char_result.num_detections << endl;
            }

            vector<JHDeepCore::Detection> valid_char_dets;
            vector<string> char_texts;

            for (int ci = 0; ci < char_result.num_detections; ci++) {
                auto& ch = char_result.detections[ci];
                Rect ch_roi = safeROI(
                    ch.bbox.x, ch.bbox.y,
                    ch.bbox.width, ch.bbox.height,
                    roi_img.cols, roi_img.rows);

                if (ch_roi.area() <= 0) continue;

                Mat char_img = roi_img(ch_roi).clone();
                if (char_img.empty()) continue;

                Mat char_img_bgr;
                if (char_img.channels() == 1) {
                    cvtColor(char_img, char_img_bgr, COLOR_GRAY2BGR);
                } else {
                    char_img_bgr = char_img;
                }

                JHDeepCore::OCRResult ocr_result = g_ocr->Recognize(char_img_bgr);

                string char_text;
                if (!ocr_result.boxes.empty()) {
                    char_text = ocr_result.boxes[0].text;
                }

                if (verbose) {
                    cout << "    char[" << ci << "] text=\""
                         << char_text << "\" conf=" << ch.confidence << endl;
                }

                valid_char_dets.push_back(ch);
                char_texts.push_back(char_text);
            }

            cout << "[DEBUG] valid_char_dets.size()=" << valid_char_dets.size()
                 << ", char_texts.size()=" << char_texts.size() << endl;
            for (int i = 0; i < (int)char_texts.size(); i++) {
                cout << "[DEBUG]   char[" << i << "] text=\"" << char_texts[i]
                     << "\" bbox=(" << valid_char_dets[i].bbox.x << ","
                     << valid_char_dets[i].bbox.y << ","
                     << valid_char_dets[i].bbox.width << ","
                     << valid_char_dets[i].bbox.height
                     << ") conf=" << valid_char_dets[i].confidence << endl;
            }

            string line_result = sortCharsByPosition(valid_char_dets, char_texts);

            cout << "[DEBUG] line_result=\"" << line_result << "\"" << endl;

            if (!line_result.empty()) {
                if (!ocr_combined.empty()) ocr_combined += ",";
                ocr_combined += line_result;
                cout << "[DEBUG] ocr_combined=\"" << ocr_combined << "\"" << endl;
            }
        }
    }

    auto infer_end = chrono::high_resolution_clock::now();
    auto infer_ms = chrono::duration_cast<chrono::milliseconds>(infer_end - infer_start).count();
    if (verbose) {
        cout << "inference time: " << infer_ms << " ms" << endl;
    }

    time_t currtime = time(NULL);
    tm* t = localtime(&currtime);

    string folderPath = format("%s\\station_%02d\\%d%02d%02d",
        g_config.result_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    ensureDirectoryExists(folderPath);
    createSplitDirectories(station_id, t);

    string save_picture_name = format("%s\\station_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d_%d.jpg",
        g_config.result_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    string save_picture_name_split_ok = format("%s\\station_%02d\\%d%02d%02d\\penma\\ok\\%d%02d%02d%02d%02d%02d_%d.jpg",
        g_config.split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    string save_picture_name_split_ng = format("%s\\station_%02d\\%d%02d%02d\\penma\\ng\\%d%02d%02d%02d%02d%02d_%d.jpg",
        g_config.split_dir.c_str(), station_id,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, pic_number);

    string dd;

    if (!ocr_combined.empty()) {
        bool isOK = false;
        string ocr_no_hash;
        for (auto c : ocr_combined) {
            if (c != '#') ocr_no_hash += c;
        }
        if ((int)ocr_no_hash.length() >= 8) {
            string first8 = ocr_no_hash.substr(0, 8);
            if (first8 == heat_number) {
                isOK = true;
                if (verbose) cout << "luhao are equal." << endl;
            }
        }

        if (isOK) {
            imwrite(save_picture_name_split_ok, src_img);
        } else {
            imwrite(save_picture_name_split_ng, src_img);
        }

        imwrite(save_picture_name, src_img);

        dd = to_string(station_id) + ";OK;Penma;" + ocr_combined + ";" + save_picture_name;
        success_flag = true;

        item["picture_id"] = pic_number + 1;
        item["state_flag"] = "OK";
        item["zifu_type"] = "Penma";
        item["result"] = ocr_combined;
        item["picture_path"] = save_picture_name;
    } else {
        imwrite(save_picture_name, src_img);
        imwrite(save_picture_name_split_ng, src_img);

        dd = to_string(station_id) + ";NG;Penma;" + save_picture_name;
        success_flag = false;

        item["picture_id"] = pic_number + 1;
        item["state_flag"] = "NG";
        item["zifu_type"] = "Penma";
        item["result"] = "";
        item["picture_path"] = save_picture_name;
    }

    if (verbose) {
        cout << (success_flag ? ">>> OK" : ">>> NG") << " " << dd << endl;
    }

    return item;
}

static json runRecognition(
    const vector<string>& picture_path_array,
    int station_id,
    const string& heat_number,
    bool verbose,
    ofstream* pfout = nullptr)
{
    json root_all;
    root_all["station_id"] = station_id;
    json array_result = json::array();

    for (int pic_number = 0; pic_number < (int)picture_path_array.size(); pic_number++)
    {
        Mat src_img = imread(picture_path_array[pic_number]);

        if (!src_img.data) {
            cerr << "empty image: " << picture_path_array[pic_number] << endl;
            json item;
            item["picture_id"] = pic_number + 1;
            item["state_flag"] = "NG";
            item["zifu_type"] = "";
            item["result"] = "";
            item["picture_path"] = "";
            array_result.push_back(item);

            if (pfout) {
                writeLog(*pfout, "detect failed! empty image");
                *pfout << "****************************************" << endl;
            }
            continue;
        }

        json item = runRecognitionSingle(src_img, station_id, heat_number,
                                          pic_number, verbose);

        array_result.push_back(item);
    }

    root_all["all_results"] = array_result;
    return root_all;
}

void handle_character_recognition(const httplib::Request& req, httplib::Response& res)
{
    lock_guard<mutex> lock(g_mutex);

    time_t currentTime = time(NULL);
    char chCurrentTime[256];
    strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
    string stCurrentTime = chCurrentTime;
    string filename = stCurrentTime + "@OCR.txt";
    string fileFullName = g_config.log_dir + "\\" + filename;
    ensureDirectoryExists(g_config.log_dir);
    ofstream fout;
    fout.open(fileFullName.c_str(), ios::app);

    int station_id = 0;
    string heat_number;
    string picture_path;

    try {
        json req_json = json::parse(req.body);
        heat_number = req_json.value("heat_number", "");
        station_id = req_json.value("station_id", 0);
        picture_path = req_json.value("picture_path", "");
    } catch (...) {
        fout << "Failed to parse request body" << endl;
    }

    fout << "recv station:" << station_id
         << " heat:" << heat_number
         << " path:" << picture_path << endl;

    vector<string> picture_path_array = splitStringByCsharp(picture_path);

    json root_all = runRecognition(picture_path_array, station_id, heat_number,
                                    false, &fout);

    res.set_content(root_all.dump(), "application/json");
    fout.close();
}

static int runLocalTest(
    const string& image_path,
    const string& heat_number,
    int station_id)
{
    cout << "=== Local Test Mode ===" << endl;
    cout << "  image:   " << image_path << endl;
    cout << "  heat:    " << heat_number << endl;
    cout << "  station: " << station_id << endl;
    cout << "  device:  " << g_config.device << endl;
    cout << endl;

    vector<string> picture_path_array = splitStringByCsharp(image_path);

    auto start = chrono::high_resolution_clock::now();
    json result = runRecognition(picture_path_array, station_id, heat_number, true);
    auto end = chrono::high_resolution_clock::now();
    auto total_ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << endl;
    cout << "=== Result ===" << endl;
    cout << result.dump(2) << endl;
    cout << "total time: " << total_ms << " ms" << endl;

    return 0;
}

static void printHelp(const char* progName)
{
    cout << "Usage: " << progName << " [options]" << endl;
    cout << endl;
    cout << "Server mode (default):" << endl;
    cout << "  " << progName << " -c config.yaml" << endl;
    cout << endl;
    cout << "Local test mode:" << endl;
    cout << "  " << progName << " --test -c config.yaml -i image.jpg -H heat -s station" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "  -c, --config   Path to config YAML file (default: config.yaml)" << endl;
    cout << "  --test         Run in local test mode (single image inference)" << endl;
    cout << "  -i, --image    Image path for local test (use # to separate multiple)" << endl;
    cout << "  -H, --heat     Heat number for local test" << endl;
    cout << "  -s, --station  Station ID for local test" << endl;
    cout << "  -v, --verbose  Verbose output for local test" << endl;
    cout << "  -h, --help     Show this help" << endl;
}

int main(int argc, char* argv[])
{
    string config_path = "config.yaml";
    bool test_mode = false;
    string test_image;
    string test_heat = "";
    int test_station = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--test") {
            test_mode = true;
        } else if ((arg == "-i" || arg == "--image") && i + 1 < argc) {
            test_image = argv[++i];
        } else if ((arg == "-H" || arg == "--heat") && i + 1 < argc) {
            test_heat = argv[++i];
        } else if ((arg == "-s" || arg == "--station") && i + 1 < argc) {
            test_station = stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        }
    }

    if (test_mode && test_image.empty()) {
        cerr << "[ERROR] --test mode requires -i <image_path>" << endl;
        printHelp(argv[0]);
        return 1;
    }

    try {
        g_config = loadConfig(config_path);
    } catch (const exception& e) {
        cerr << "[ERROR] Failed to load config: " << e.what() << endl;
        return 1;
    }

    if (!test_mode) {
        cout << "=== dabang_jiguang OCR HTTP service ===" << endl;
    }
    cout << "[INFO] Config: " << config_path << endl;
    if (!test_mode) {
        cout << "[INFO] Server: " << g_config.host << ":" << g_config.port << endl;
    }
    cout << "[INFO] Device: " << g_config.device << endl;

    auto start_time = chrono::high_resolution_clock::now();

    try {
        g_det_label = new JHDeepCore::Detector(g_config.label_detect_model, g_config.device);
        cout << "[OK] Label detect model loaded: " << g_config.label_detect_model << endl;

        g_det_char = new JHDeepCore::Detector(g_config.char_detect_model, g_config.device);
        cout << "[OK] Char detect model loaded: " << g_config.char_detect_model << endl;

        JHDeepCore::OCRRecognizer::Params ocr_params;
        ocr_params.rec_model_path = g_config.ocr_rec_model;
        ocr_params.rec_label_path = g_config.ocr_rec_label;
        ocr_params.device = g_config.device;
        g_ocr = new JHDeepCore::OCRRecognizer(ocr_params);
        cout << "[OK] OCR model loaded: " << g_config.ocr_rec_model << endl;
    } catch (const exception& e) {
        cerr << "[ERROR] Failed to initialize models: " << e.what() << endl;
        return 1;
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "[INFO] Model init time: " << duration.count() << " ms" << endl;

    if (test_mode) {
        int ret = runLocalTest(test_image, test_heat, test_station);
        delete g_ocr;
        delete g_det_char;
        delete g_det_label;
        return ret;
    }

    httplib::Server server;

    server.Get("/hello", [](const httplib::Request&, httplib::Response& resp) {
        resp.set_content("Hello World! dabang_jiguang OCR service running.", "text/plain");
        resp.status = 200;
    });

    server.Post("/character_recognition", handle_character_recognition);

    server.Get("/stop", [&](const httplib::Request&, httplib::Response&) {
        cout << "[INFO] Received stop signal." << endl;
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

    cout << "[INFO] HTTP server start: " << g_config.host << ":" << g_config.port << endl;
    cout << "[INFO] POST /character_recognition" << endl;
    cout << "[INFO] GET  /hello" << endl;
    cout << "[INFO] GET  /stop" << endl;

    if (!server.listen(g_config.host, g_config.port)) {
        cerr << "[ERROR] Failed to start server on " << g_config.host << ":" << g_config.port << endl;
        delete g_ocr;
        delete g_det_char;
        delete g_det_label;
        return 1;
    }

    delete g_ocr;
    delete g_det_char;
    delete g_det_label;
    g_ocr = nullptr;
    g_det_char = nullptr;
    g_det_label = nullptr;

    cout << "[INFO] Server stopped." << endl;
    return 0;
}
