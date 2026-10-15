#include "reel_debug_logger.h"

#include "file_utils.h"
#include "json.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using ojson = nlohmann::ordered_json;

namespace JHDeepCore {
namespace Pipeline {

namespace {

std::string formatTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm lt_tm{};
    localtime_s(&lt_tm, &t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &lt_tm);
    std::ostringstream out;
    out << buf << '.' << std::setw(3) << std::setfill('0') << ms.count();
    return out.str();
}

std::string todayString()
{
    std::time_t t = std::time(nullptr);
    std::tm lt_tm{};
    localtime_s(&lt_tm, &t);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &lt_tm);
    return std::string(buf);
}

ojson rectToJson(const ReelRect &r)
{
    ojson j;
    j["x"] = r.x;
    j["y"] = r.y;
    j["w"] = r.width;
    j["h"] = r.height;
    j["type"] = r.type;
    return j;
}

ojson rectsToJson(const std::vector<ReelRect> &rects)
{
    ojson arr = ojson::array();
    for (const auto &r : rects) arr.push_back(rectToJson(r));
    return arr;
}

ojson pointToJson(const ReelPoint &p)
{
    ojson j;
    j["x"] = p.x;
    j["y"] = p.y;
    return j;
}

ojson dedupToJson(const ReelDedupDecision &d)
{
    ojson j;
    j["evaluated"] = d.evaluated;
    j["applied"] = d.applied;
    j["multi_reel_exempt"] = d.multi_reel_exempt;
    j["removed_count"] = d.removed_count;
    j["prev_camera_id"] = d.prev_camera_id;
    j["prev_inside_count"] = d.prev_inside_count;
    j["has_prev_inside"] = d.has_prev_inside;
    j["has_present"] = d.has_present;
    if (d.has_prev_inside) j["prev_last_center"] = rectToJson(d.prev_last_center);
    if (d.has_present) j["present_first_center"] = rectToJson(d.present_first_center);
    j["has_prev_qishi"] = d.has_prev_qishi;
    j["has_present_qishi"] = d.has_present_qishi;
    if (d.has_prev_qishi) j["prev_qishi_end"] = pointToJson(d.prev_qishi_end);
    if (d.has_present_qishi) {
        j["present_qishi_start"] = pointToJson(d.present_qishi_start);
    }
    j["prev_distance"] = d.prev_distance;
    j["present_distance"] = d.present_distance;
    return j;
}

} // namespace

ReelDebugLogger::ReelDebugLogger(bool enabled, const std::string &log_dir)
    : enabled_(enabled), log_dir_(log_dir)
{
}

void ReelDebugLogger::ensureOpenForToday()
{
    std::string date = todayString();
    if (stream_.is_open() && date == current_date_) return;
    if (stream_.is_open()) stream_.close();
    FileHelper::ensureDirectoryExists(log_dir_);
    std::string path = log_dir_ + "\\tedai_debug_" + date + ".jsonl";
    stream_.open(path.c_str(), std::ios::app);
    current_date_ = date;
}

void ReelDebugLogger::logCamera(const std::string &station_id, uint64_t req_index,
                                int read_picture_flag,
                                const ReelCameraDiagnostics &diag)
{
    if (!enabled_) return;
    ensureOpenForToday();
    if (!stream_.is_open()) return;

    ojson j;
    j["req_index"] = req_index;
    j["ts"] = formatTimestamp();
    j["station_id"] = station_id;
    j["camera_id"] = diag.camera_id;
    j["read_picture_flag"] = read_picture_flag;
    j["det_ok"] = diag.det_ok;

    ojson raw = ojson::array();
    for (const auto &o : diag.raw) {
        ojson r;
        r["cx"] = o.rect.x;
        r["cy"] = o.rect.y;
        r["w"] = o.rect.width;
        r["h"] = o.rect.height;
        r["classid"] = o.classid;
        r["prob"] = o.prob;
        raw.push_back(r);
    }
    j["raw"] = raw;

    ojson assigned;
    assigned["inside"] = rectsToJson(diag.inside_assigned);
    assigned["outside"] = rectsToJson(diag.outside_assigned);
    assigned["collect"] = rectsToJson(diag.collect_assigned);
    j["assigned"] = assigned;

    ojson dedup;
    dedup["inside_count_before"] = diag.inside_count_before_dedup;
    dedup["inside_count_after"] = diag.inside_count_after_dedup;
    dedup["guolv_flag"] = diag.guolv_flag;
    dedup["guolv_flag_outside"] = diag.guolv_flag_outside;
    dedup["inside"] = dedupToJson(diag.dedup_inside);
    dedup["outside"] = dedupToJson(diag.dedup_outside);
    j["dedup"] = dedup;

    ojson final_j;
    final_j["inside"] = diag.final_inside;
    final_j["outside"] = diag.final_outside;
    final_j["collect"] = diag.final_collect;
    j["final"] = final_j;

    stream_ << j.dump() << '\n';
    stream_.flush();
}

} // namespace Pipeline
} // namespace JHDeepCore
