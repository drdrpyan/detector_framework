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
  void Detect(const cv::Mat& img, bool do_nms,
              std::vector<DetectionT>* result);
  void Detect(const std::vector<cv::Mat>& imgs, bool do_nms,
              std::vector<std::vector<DetectionT> >* result);
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
  virtual void Detect_impl(const cv::Mat& img, 
                           std::vector<DetectionT>* result) = 0;
  virtual void Detect_impl(
      const std::vector<cv::Mat>& imgs,
      std::vector<std::vector<DetectionT> >* result) = 0;

 private:
  std::shared_ptr<NMS<DetectionT> > nms_;
};

// template fucntions
template <typename DetectionT>
void Detector<DetectionT>::Detect(
    const cv::Mat& img, bool do_nms, std::vector<DetectionT>* result) {
  assert(result);

  if (do_nms && nms_ != nullptr) {
    std::vector<DetectionT> temp_result;
    Detect_impl(img, &temp_result);
    nms_->nms(temp_result, result);
  }
  else {
    Detect_impl(img, result);
  }
}

template <typename DetectionT>
void Detector<DetectionT>::Detect(
    const std::vector<cv::Mat>& imgs, bool do_nms,
    std::vector<std::vector<DetectionT> >* result) {
  assert(result);

  auto sub_iter = out_beg->begin();

  if (do_nms && nms_ != nullptr) {
    std::vector<std::vector<DetectionT> > temp_result;
    Detect_impl(imgs, temp_result);
    
    result->resize(imgs.size());
    for (int i = 0; i < temp_result.size(); ++i) {
      nms_->nms(temp_result[i], &((*result)[i]));
    }
  }
  else {
    Detect_impl(imgs, result);
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
