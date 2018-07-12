#ifndef BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_
#define BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_

#include "detection_filter.hpp"

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
  Detector(bool do_filtering = false,
           DetectionFilter<DetectionT>* filter = nullptr);

  template <typename OutIterT>
  void Detect(const cv::Mat& img, OutIterT& result_beg);

  template <typename OutIterT>
  void Detect(const cv::Mat& img, bool do_filtering, OutIterT& result_beg);

  template <typename InIterT, typename OutIterT>
  void Detect(const InIterT& img_beg, const InIterT& img_end,
              OutIterT& result_beg);

  template <typename InIterT, typename OutIterT>
  void Detect(const InIterT& img_beg, const InIterT& img_end,
              bool do_nms, OutIterT& result_beg);

  void Detect(const cv::Mat& img, bool do_filtering,
              std::vector<DetectionT>* result);
  void Detect(const std::vector<cv::Mat>& imgs, bool do_filtering,
              std::vector<std::vector<DetectionT> >* result);

  void set_do_filtering(bool on);
  void set_filter(std::shared_ptr<DetectionFilter<DetectionT> >& filter);
  void set_filter(DetectionFilter<DetectionT>* filter);

 protected:
  virtual void Detect_impl(const cv::Mat& img,
                           std::vector<DetectionT>* result) = 0;
  virtual void Detect_impl(
    const std::vector<cv::Mat>& imgs,
    std::vector<std::vector<DetectionT> >* result) = 0;

 private:
  bool do_filtering_;
  std::shared_ptr<DetectionFilter<DetectionT> > filter_;
};

// template fucntions
template <typename DetectionT>
inline Detector<DetectionT>::Detector(
    bool do_filtering, DetectionFilter<DetectionT>* filter) 
  : do_filtering_(do_filtering), filter_(filter) {

}


template <typename DetectionT>
template <typename OutIterT>
inline void Detector<DetectionT>::Detect(const cv::Mat& img,
                                         OutIterT& result_beg) {
  Detect(img, do_filtering_, result_beg);
}

template <typename DetectionT>
template <typename OutIterT>
inline void Detector<DetectionT>::Detect(const cv::Mat& img, 
                                         bool do_filtering,
                                         OutIterT& result_beg) {
  std::vector<DetectionT> temp_result;
  Detect(img, do_filtering, &temp_result);
  std::copy(temp_result.cbegin(), temp_result.cend(), result_beg);
}

template <typename DetectionT>
template <typename InIterT, typename OutIterT>
inline void Detector<DetectionT>::Detect(const InIterT& img_beg,
                                         const InIterT& img_end,
                                         OutIterT& result_beg) {
  Detect(img_beg, img_end, do_filtering_, result_beg);
}

template <typename DetectionT>
template <typename InIterT, typename OutIterT>
inline void Detector<DetectionT>::Detect(const InIterT& img_beg,
                                         const InIterT& img_end,
                                         bool do_filtering, 
                                         OutIterT& result_beg) {
  std::vector<cv::Mat> temp_in(img_beg, img_end);
  std::vector<std::vector<DetectionT> > temp_out;
  Detect(temp_in, do_filtering, &temp_out);
  std::copy(temp_out.cbegin(), temp_out.cend(), result_beg);
}

template <typename DetectionT>
void Detector<DetectionT>::Detect(
    const cv::Mat& img, bool do_filtering, 
    std::vector<DetectionT>* result) {
  assert(result);

  if (do_filtering && filter_ != nullptr) {
    std::vector<DetectionT> temp_result;
    Detect_impl(img, &temp_result);
    filter_->Filter(temp_result, result);
  }
  else {
    Detect_impl(img, result);
  }
}

template <typename DetectionT>
void Detector<DetectionT>::Detect(
    const std::vector<cv::Mat>& imgs, bool do_filtering,
    std::vector<std::vector<DetectionT> >* result) {
  assert(result);

  if (do_filtering && filter_ != nullptr) {
    std::vector<std::vector<DetectionT> > temp_result;
    Detect_impl(imgs, &temp_result);
    
    result->resize(imgs.size());
    for (int i = 0; i < temp_result.size(); ++i) {
      filter_->Filter(temp_result[i], &((*result)[i]));
    }
  }
  else {
    Detect_impl(imgs, result);
  }
}

// inline functions
template <typename DetectionT>
inline void Detector<DetectionT>::set_do_filtering(bool on) {
  do_filtering_ = on;
}

template <typename DetectionT>
inline void Detector<DetectionT>::set_filter(
    std::shared_ptr<DetectionFilter<DetectionT> >& filter) {
  filter_ = filter;
}

template <typename DetectionT>
inline void Detector<DetectionT>::set_filter(
    DetectionFilter<DetectionT>* filter) {
  filter_->reset(filter);
}


} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_DETECTOR_HPP_
