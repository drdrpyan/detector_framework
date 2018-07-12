#ifndef BGM_DETECTION_FRAMEWORK_DETECTOR_FOR_ROI_HPP_
#define BGM_DETECTION_FRAMEWORK_DETECTOR_FOR_ROI_HPP_

#include "detector.hpp"

#include <opencv2/core.hpp>

#include <glog/logging.h>

#include <vector>

namespace bgm
{

template <typename DetectionT, typename ROICoordT=int>
class DetectorForROI : public Detector<DetectionT>
{
 public:
  template <typename InIterT>
  DetectorForROI(Detector<DetectionT>* core_detector,
                 const InIterT& roi_beg, const InIterT& roi_end,
                 DetectionFilter<DetectionT>* filter = nullptr,
                 bool do_filtering = false);
  template <typename InIterT>
  void set_roi(const InIterT& rect_beg, const InIterT& rect_end);

 private:
  virtual void Detect_impl(const cv::Mat& img, 
                           std::vector<DetectionT>* result) override;
  virtual void Detect_impl(
      const std::vector<cv::Mat>& imgs,
      std::vector<std::vector<DetectionT> >* result) override;

  template <typename OutIterT>
  void ExtractPatches(const cv::Mat& in_mat,
                      OutIterT& patches_beg) const;
  template <typename InOutIterT>
  void MovePatchResults(InOutIterT& patch_results_beg,
                        InOutIterT& patch_results_end) const;

  std::shared_ptr<Detector<DetectionT> > core_detector_;
  std::vector<cv::Rect_<ROICoordT> > roi_;

}; // class DetectorForROI

// template functions
template <typename DetectionT, typename ROICoordT>
template <typename InIterT>
DetectorForROI<DetectionT, ROICoordT>::DetectorForROI(
    Detector<DetectionT>* core_detector, 
    const InIterT& roi_beg, const InIterT& roi_end,
    DetectionFilter<DetectionT>* filter, bool do_filtering)
  : Detector<DetectionT>(filter, do_filtering),
    core_detector_(core_detector), roi_(roi_beg, roi_end) {

}


template <typename DetectionT, typename ROICoordT>
template <typename InIterT>
void DetectorForROI<DetectionT, ROICoordT>::set_roi(
    const InIterT& rect_beg, const InIterT& rect_end) {
  roi_.assign(rect_beg, rect_end);
}

template <typename DetectionT, typename ROICoordT>
inline void DetectorForROI<DetectionT, ROICoordT>::Detect_impl(
    const cv::Mat& img, std::vector<DetectionT>* result) {
  CHECK(result);
  
  std::vector<cv::Mat> patches;
  ExtractPatches(img, std::back_inserter(patches));

  core_detector_->Detect(patches, result);
}

template <typename DetectionT, typename ROICoordT>
void DetectorForROI<DetectionT, ROICoordT>::Detect_impl(
    const std::vector<cv::Mat>& imgs,
    std::vector<std::vector<DetectionT> >* result) {
  CHECK_GT(imgs.size(), 0);
  CHECK(result);

  std::vector<std::vector<cv::Mat> > patches(img.size());
  for (int i = 0; i < patches.size(); ++i)
    ExtractPatches(imgs[i], std::back_inserter(patches[i]));
  
  core_detector_->Detect(patches, result);
}

template <typename DetectionT, typename ROICoordT>
template <typename OutIterT>
inline void DetectorForROI<DetectionT, ROICoordT>::ExtractPatches(
    const cv::Mat& in_mat, OutIterT& patches_beg) const {
  CHECK_GT(roi_.size, 0);

  OutIterT out_iter = patches_beg;
  for (int i = 0; i < roi_.size(); ++i)
    *(out_iter++) = in_mat(roi_[i]);
}

template <typename DetectionT, typename ROICoordT>
template <typename InOutIterT>
void DetectorForROI<DetectionT, ROICoordT>::MovePatchResults(
    InOutIterT& patch_results_beg, InOutIterT& patch_results_end) const {
  CHECK_GT(std::distance(patch_results_beg, patch_results_end), 
           roi_.size());
  
  InOutIterT patch_iter = patch_results_beg;
  for (int i = 0; i < roi_.size(); ++i) {
    auto elem_iter = patch_iter->begin();
    auto elem_end = patch_iter->end();
    while (elem_iter != elem_end) {
      auto rect = elem_iter->rect();
      rect.x += roi_[i].x;
      rect.y += roi_[i].y;
      elem_iter->set_rect(rect);
      ++elem_iter;
    }
    ++patch_iter;
  }
}

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_DETECTOR_FOR_ROI_HPP_
