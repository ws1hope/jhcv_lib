#include "JHDeepCore.h"
#include "section_angle_internal.h"

namespace JHDeepCore {

class SectionAngleCheckerPrivate {
public:
    SectionAngleCheckerPrivate(const std::string &model_path,
                               const section_angle::SectionAngleConfig &config,
                               const std::string &label_path,
                               int device_id,
                               const std::string &config_path)
        : config_(config),
          segmenter_(model_path, label_path, device_id, config_path)
    {
    }

    void process(const cv::Mat &image, std::vector<SectionAngleItem> &results)
    {
        results.clear();
        if (image.empty()) {
            return;
        }

        std::vector<cv::Mat> images = {image};
        std::vector<SegmentationResult> seg_results;
        segmenter_.process(images, seg_results);
        if (seg_results.empty() || seg_results[0].segmentation_mask.empty()) {
            return;
        }

        cv::Mat class_mask = section_angle::extractClassMask(
            seg_results[0], config_.target_class_id);
        if (cv::countNonZero(class_mask) <= 0) {
            return;
        }

        std::vector<section_angle::ContourAngleDetail> details = section_angle::analyzeMask(
            class_mask, image.size(), config_);

        results.reserve(details.size());
        for (const auto &detail : details) {
            if (detail.instance_id <= 0) {
                continue;
            }
            results.push_back(section_angle::toPublicItem(detail));
        }
    }

private:
    section_angle::SectionAngleConfig config_;
    Segmenter segmenter_;
};

SectionAngleChecker::SectionAngleChecker(const std::string &model_path,
                                         int target_class_id,
                                         float angle_tolerance_deg,
                                         const std::string &label_path,
                                         int device_id,
                                         const std::string &config_path)
{
    section_angle::SectionAngleConfig config;
    config.target_class_id = target_class_id;
    config.angle_tolerance_deg = angle_tolerance_deg;
    m_pHandle = std::make_shared<SectionAngleCheckerPrivate>(
        model_path, config, label_path, device_id, config_path);
}

SectionAngleChecker::~SectionAngleChecker() = default;

void SectionAngleChecker::process(const cv::Mat &image,
                                  std::vector<SectionAngleItem> &results)
{
    m_pHandle->process(image, results);
}

} // namespace JHDeepCore
