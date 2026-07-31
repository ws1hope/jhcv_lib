#include "JHDeepCore.h"
#include "zbsltj_pipeline.h"
#include "file_utils.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::json;

namespace JHDeepCore {

static std::vector<std::string> splitByComma(const std::string& str)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string segment;
    while (std::getline(ss, segment, ',')) {
        result.push_back(segment);
    }
    return result;
}

class ZbsltjServicePrivate {
public:
    explicit ZbsltjServicePrivate(const std::string& config_path)
        : config_(FileHelper::loadZbsltjConfig(config_path))
    {
        ZbsltjConfig pcfg;
        pcfg.billet_det_model = config_.billet_det_model;
        pcfg.char_seg_model = config_.char_seg_model;
        pcfg.ocr_model = config_.ocr_model;
        pcfg.ocr_label = config_.ocr_label;
        pcfg.device = config_.device;

        pipeline_ = std::make_unique<Pipeline::ZbsltjPipeline>(pcfg);
    }

    const ZbsltjServerConfig& config() const { return config_; }

    std::string handleRequest(const std::string& req_body)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        auto req_start = std::chrono::high_resolution_clock::now();

        // open daily log file
        time_t currentTime = time(NULL);
        char chCurrentTime[256];
        strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d", localtime(&currentTime));
        std::string filename = std::string(chCurrentTime) + "@Zbsltj.txt";
        std::string fileFullName = config_.log_dir + "\\" + filename;
        FileHelper::ensureDirectoryExists(config_.log_dir);
        std::ofstream fout;
        fout.open(fileFullName.c_str(), std::ios::app);

        fout << "Request received." << std::endl;

        int first_seq = 0;
        std::string heat_number;
        std::string pdi_count;
        std::string lotno_number;

        struct PicInfo {
            int camera_id;
            std::string picture_info;
        };
        std::vector<PicInfo> pictures;

        try {
            json req_json = json::parse(req_body);
            first_seq = req_json.value("first_seq", 0);
            heat_number = req_json.value("heat_number", "");
            pdi_count = req_json.value("pdi_count", "");
            lotno_number = req_json.value("lotno_number", "");

            if (req_json.contains("pictures_information") && req_json["pictures_information"].is_array()) {
                for (auto& item : req_json["pictures_information"]) {
                    PicInfo pi;
                    pi.camera_id = item.value("camera_id", -1);
                    pi.picture_info = item.value("picture_info", "");
                    pictures.push_back(pi);
                }
            }
        } catch (...) {
            fout << "Failed to parse request body" << std::endl;
        }

        // parse candidate heats and pdis
        auto candidate_heats = splitByComma(heat_number);
        auto candidate_pdis = splitByComma(pdi_count);

        // init seq from first_seq if not yet set
        if (first_seq > 0) {
            if (cam0_seq_ == 0) cam0_seq_ = first_seq;
            if (cam1_seq_ == 0) cam1_seq_ = first_seq;
        }

        // check for duplicate camera_id
        std::vector<int> cam_ids;
        for (auto& p : pictures) cam_ids.push_back(p.camera_id);
        std::sort(cam_ids.begin(), cam_ids.end());
        bool dup_id = false;
        for (size_t i = 1; i < cam_ids.size(); i++) {
            if (cam_ids[i] == cam_ids[i - 1]) { dup_id = true; break; }
        }

        json root_all;
        json array_all_results = json::array();
        json array_piliao_count = json::array();
        int current_billet_count = 0;

        if (pictures.empty()) {
            fout << "Received empty pictures_information" << std::endl;
        } else if (dup_id) {
            fout << "Duplicate camera_id detected" << std::endl;
        } else {
            // process each picture
            for (auto& pic : pictures) {
                int cam_id = pic.camera_id;
                std::string img_path = pic.picture_info;

                fout << "camera_id=" << cam_id << " path=" << img_path << std::endl;

                cv::Mat src_img = cv::imread(img_path);
                if (src_img.empty()) {
                    fout << "Failed to read image: " << img_path << std::endl;
                    json item;
                    item["camera_id"] = cam_id;
                    item["picture_info"] = img_path;
                    item["rec_results"] = "";
                    item["zhengfan"] = "NG";
                    array_all_results.push_back(item);
                    continue;
                }

                // get per-camera state
                std::string& cur_heat = (cam_id == 0) ? cam0_current_heat_ : cam1_current_heat_;
                int& cur_seq = (cam_id == 0) ? cam0_seq_ : cam1_seq_;

                // run pipeline
                auto results = pipeline_->process(
                    src_img, candidate_heats, candidate_pdis,
                    cur_heat, cur_seq, true);

                // sync heat between cameras (cam1 follows cam0)
                if (cam_id == 0) {
                    cam1_current_heat_ = cam0_current_heat_;
                }

                current_billet_count += results.size();

                // per-step timing + per-billet results
                const auto& T = pipeline_->lastTiming();
                fout << "[timing] cam=" << cam_id
                     << " det=" << T.det_ms << "ms"
                     << " seg=" << T.seg_ms << "ms"
                     << " ocr=" << T.ocr_ms << "ms"
                     << " total=" << T.total_ms << "ms"
                     << " billets=" << results.size() << std::endl;
                for (size_t i = 0; i < results.size(); i++) {
                    std::string texts_joined;
                    for (size_t k = 0; k < results[i].rec_texts.size(); k++) {
                        if (k > 0) texts_joined += "#";
                        texts_joined += results[i].rec_texts[k];
                    }
                    if (results[i].rec_texts.empty()) texts_joined = "0";
                    fout << "[billet] cam=" << cam_id << " #" << (i + 1)
                         << " seq=" << results[i].seq_number
                         << " heat=\"" << results[i].matched_heat << "\""
                         << " pdi=\"" << results[i].pdi_count << "\""
                         << " ocr=\"" << texts_joined << "\""
                         << " ok=" << (results[i].success ? 1 : 0) << std::endl;
                }

                // draw results
                pipeline_->drawResults(src_img, results);

                // save result image
                time_t currtime = time(NULL);
                tm* t = localtime(&currtime);
                std::string folderPath = cv::format("%s\\camera_%02d\\%d%02d%02d",
                    config_.result_dir.c_str(), cam_id,
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
                FileHelper::ensureDirectoryExists(folderPath);

#ifdef _WIN32
                SYSTEMTIME st;
                GetLocalTime(&st);
                std::string save_name = cv::format(
                    "%s\\camera_%02d\\%d%02d%02d\\%d%02d%02d%02d%02d%02d%03d_%s.jpg",
                    config_.result_dir.c_str(), cam_id,
                    st.wYear, st.wMonth, st.wDay,
                    st.wYear, st.wMonth, st.wDay,
                    st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
                    heat_number.c_str());
#else
                std::string save_name = cv::format(
                    "%s/camera_%02d/%d%02d%02d/%d%02d%02d%02d%02d%02d_%s.jpg",
                    config_.result_dir.c_str(), cam_id,
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                    t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                    t->tm_hour, t->tm_min, t->tm_sec, heat_number.c_str());
#endif
                cv::imwrite(save_name, src_img);

                // build rec_results string
                std::string combined_out;
                for (auto& r : results) {
                    std::string result_ocr;
                    for (int k = 0; k < (int)r.rec_texts.size(); k++) {
                        if (k > 0) result_ocr += "#";
                        result_ocr += r.rec_texts[k];
                    }
                    if (r.rec_texts.empty()) result_ocr += "0";

                    std::string new_output = Pipeline::ZbsltjPipeline::formatPenmaResult(result_ocr);
                    new_output.erase(std::remove(new_output.begin(), new_output.end(), ' '),
                                     new_output.end());
                    if (!combined_out.empty()) combined_out += ",";
                    combined_out += new_output;
                }

                // determine zhengfan
                std::string zhengfan = "OK";

                json item;
                item["camera_id"] = cam_id;
                item["picture_info"] = save_name;
                item["rec_results"] = combined_out;
                item["zhengfan"] = zhengfan;
                array_all_results.push_back(item);
            }
        }

        root_all["all_results"] = array_all_results;

        json piliao_item;
        piliao_item["piliao_count"] = current_billet_count;
        array_piliao_count.push_back(piliao_item);
        root_all["piliao_count"] = array_piliao_count;

        auto req_end = std::chrono::high_resolution_clock::now();
        double req_ms = std::chrono::duration<double, std::milli>(req_end - req_start).count();
        fout << "Request processed. total=" << req_ms << "ms" << std::endl;
        fout.close();

        std::cout << "[RESULT] " << root_all.dump() << std::endl;
        return root_all.dump();
    }

    int runLocalTest(const std::string& image_path,
                      const std::string& heat_number,
                      int camera_id)
    {
        std::cout << "=== Zbsltj Local Test ===" << std::endl;
        std::cout << "  image:   " << image_path << std::endl;
        std::cout << "  heat:    " << heat_number << std::endl;
        std::cout << "  camera:  " << camera_id << std::endl;
        std::cout << "  device:  " << config_.device << std::endl;

        cv::Mat src_img = cv::imread(image_path);
        if (src_img.empty()) {
            std::cerr << "[ERROR] Cannot read image: " << image_path << std::endl;
            return -1;
        }

        auto candidate_heats = splitByComma(heat_number);
        std::vector<std::string> candidate_pdis;

        std::string& cur_heat = (camera_id == 0) ? cam0_current_heat_ : cam1_current_heat_;
        int& cur_seq = (camera_id == 0) ? cam0_seq_ : cam1_seq_;

        auto start = std::chrono::high_resolution_clock::now();
        auto results = pipeline_->process(
            src_img, candidate_heats, candidate_pdis,
            cur_heat, cur_seq, true);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        pipeline_->drawResults(src_img, results);

        std::cout << std::endl << "=== Result ===" << std::endl;
        for (int i = 0; i < (int)results.size(); i++) {
            std::cout << "Billet " << i << ":" << std::endl;
            std::cout << "  seq:     " << results[i].seq_number << std::endl;
            std::cout << "  heat:    " << results[i].matched_heat << std::endl;
            std::cout << "  pdi:     " << results[i].pdi_count << std::endl;
            std::cout << "  texts:   ";
            for (auto& t : results[i].rec_texts) std::cout << "[" << t << "] ";
            std::cout << std::endl;
        }
        std::cout << "total time: " << total_ms << " ms" << std::endl;

        return 0;
    }

    ZbsltjServerConfig config_;
    std::unique_ptr<Pipeline::ZbsltjPipeline> pipeline_;
    std::mutex mtx_;

    // cross-request state
    std::string cam0_current_heat_;
    std::string cam1_current_heat_;
    int cam0_seq_ = 0;
    int cam1_seq_ = 0;
};

ZbsltjService::ZbsltjService(const std::string& config_path)
    : m_pHandle(std::make_shared<ZbsltjServicePrivate>(config_path))
{
}

ZbsltjService::~ZbsltjService() = default;

const ZbsltjServerConfig& ZbsltjService::config() const
{
    return m_pHandle->config();
}

std::string ZbsltjService::handleRequest(const std::string& req_body)
{
    return m_pHandle->handleRequest(req_body);
}

int ZbsltjService::runLocalTest(const std::string& image_path,
                                  const std::string& heat_number,
                                  int camera_id)
{
    return m_pHandle->runLocalTest(image_path, heat_number, camera_id);
}

} // namespace JHDeepCore
