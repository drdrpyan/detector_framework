#ifndef BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_
#define BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_

#include "nms.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <iterator>
#include <memory>
#include <vector>

namespace bgm
{

template <typename DetectionT>
class Detector
{
 public:
  template <typename OutIterT>
  void Detect(const cv::Mat& img, OutIterT& out_beg,
              bool do_nms = true);
  template <typename InIterT, typename OutIterT>
  void Detect(const InIterT& img_beg, const InIterT& img_end,
              OutIterT& out_beg, bool do_nms = true);
  //template <typename InIterT, typename OutIterT>
  //void DetectFromROIs(const cv::Mat& img,
  //                    const InIterT& roi_beg, const InIterT& roi_end,
  //                    OutIterT& out_beg, bool do_nms = true);
  //template <typename InImgIterT, typename InOffsetIterT, typename OutIterT>
  //void DetectFromROIs(const InImgIterT& img_roi_beg,
  //                    const InImgIterT& img_roi_end,
  //                    const InOffsetIterT& offset_beg,
  //                    const InOffsetIterT& offset_end,
  //                    OutIterT& out_beg, bool do_nms = true);
  void set_nms(NMS<DetectionT>* nms);

 protected:
  template <typename OutIterT>
  virtual void Detect_impl(const cv::Mat& img, OutIterT& out_beg) = 0;
  template <typename InIterT, typename OutIterT>
  virtual void Detect_impl(const InIterT& img_beg, const InIterT& img_end,
                           OutIterT& out_beg) = 0;

 private:
  std::shared_ptr<NMS<DetectionT> > nms_;
};

// template fucntions
template <typename DetectionT>
template <typename OutIterT>
void Detector<DetectionT>::Detect(const cv::Mat& img, OutIterT& out_beg,
                                  bool do_nms) {
  if (do_nms && nms_ != nullptr) {
    std::vector<DetectionT> temp_result;
    Detect_impl(img, temp_result.begin());
    nms_->nms(temp_result.begin(), temp_result.end(), out_beg);
  }
  else {
    Detect_impl(img, out_beg);
  }
}

template <typename DetectionT>
template <typename InIterT, typename OutIterT>
void Detector<DetectionT>::Detect(
    const InIterT& img_beg, const InIterT& img_end,
    OutIterT& out_beg, bool do_nms) {
  assert(std::distance(img_beg, img_end) > 0);

  auto sub_iter = out_beg->begin();

  if (do_nms && nms_ != nullptr) {
    std::vector<std::vector<DetectionT> > temp_result;
    Detect_impl(img_beg, img_end, temp_result.begin());
    
    auto sub_out_iter = out_beg->begin();
    for (auto temp_iter = temp_result.cbegin();
         temp_iter != temp_result.cend(); ++temp_iter) {
      nms_->nms(temp_iter->cbegin(), temp_iter->cend(),
                (sub_out_iter++)->begin());
    }
  }
  else {
    Detect_impl(img_beg, img_end, out_beg);
  }
}

//template <typename DetectionT>
//template <typename InIterT, typename OutIterT>
//void Detector<DetectionT>::DetectFromROIs(const cv::Mat& img,
//                                          const InIterT& roi_beg,
//                                          const InIterT& roi_end,
//                                          OutIterT& out_beg,
//                                          bool do_nms) {
//  assert(std::distance(roi_beg, roi_end) > 0);
//
//  std::vector<cv::Mat> extracted_rois;
//  std::vector<cv::Point> offsets;
//  for (auto roi_iter = roi_beg; roi_iter != roi_end; ++roi_iter) {
//    extracted_rois.push_back(img(*roi_iter));
//    offsets.push_back(roi_iter->tl());
//  }
//
//  DetectFromROIs(extracted_rois.cbegin(), extracted_rois.cend(),
//                 offsets.cbegin(), offsets.cend(), out_beg, do_nms);
//
//  
//}

//template <typename DetectionT>
//template <typename InImgIterT, typename InOffsetIterT, typename OutIterT>
//void Detector<DetectionT>::DetectFromROIs(const InImgIterT& img_roi_beg,
//                                          const InImgIterT& img_roi_end,
//                                          const InOffsetIterT& offset_beg,
//                                          const InOffsetIterT& offset_end,
//                                          OutIterT& out_beg, 
//                                          bool do_nms) {
//  int num_roi = std::distance(img_roi_beg, img_roi_end);
//  assert(num_roi > 0);
//  assert(std::distance(offset_beg, offset_end) == num_roi);
//
//  std::vector<std::vector<DetectionT> > temp_result;
//  Detect_impl(extracted_rois.cbegin(), extracted_rois.cend(),
//              temp_result.begin());
//  for()
//
//  if (do_nms && nms_ != nullptr) {
//    std::vector<DetectionT> temp_result_concat;
//    for (auto iter = temp_result.cbegin();
//         iter != temp_result.cend(); ++iter)
//      temp_result_concat.insert(iter->cbegin(), iter->cend(),
//                                temp_result_concat.end());
//    nms_->nms(temp_result_concat.cbegin(), temp_result_concat.cend(),
//              out_beg);
//  }
//  else {
//    auto out_iter = out_beg;
//    for (auto batch_iter = temp_result.cbegin();
//         batch_iter != temp_result.cend(); ++batch_iter) {
//      for (auto elem_iter = batch_iter->cbegin();
//           elem_iter != batch_iter->cend(); ++eleom_iter) {
//        *(out_iter++) = *elem_iter;
//      }
//    }
//  }
//}

// inline functions
template <typename DetectionT>
inline void Detector<DetectionT>::set_nms(NMS<DetectionT>* nms) {
  nms_->reset(nms);
}


} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_
