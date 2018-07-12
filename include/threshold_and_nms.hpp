#ifndef BGM_DETECTION_FRAMEWORK_THRESHOLD_AND_NMS_HPP_
#define BGM_DETECTION_FRAMEWORK_THRESHOLD_AND_NMS_HPP_

#include "detection_filter.hpp"

#include "nms.hpp"

#include <glog/logging.h>

#include <memory>

namespace bgm
{

template <typename DetectionT, typename ConfT = float>
class ThresholdAndNMS : public DetectionFilter<DetectionT>
{
  public:
  ThresholdAndNMS(ConfT threshold, 
                  std::shared_ptr<NMS<DetectionT> >& nms);
  ThresholdAndNMS(ConfT threshold, NMS<DetectionT>* nms);

  virtual void Filter(const std::vector<DetectionT>& src,
                      std::vector<DetectionT>* dst) override;
  void set_thrshold(ConfT threshold);
  void set_nms(std::shared_ptr<NMS<DetectionT> >& nms);
  void set_nms(NMS<DetectionT>* nms);

  private:
  ConfT threshold_;
  std::shared_ptr<NMS<DetectionT> > nms_;
}; // class ThresholdAndNMS


// template functions
template <typename DetectionT, typename ConfT>
void ThresholdAndNMS<DetectionT, ConfT>::Filter(
  const std::vector<DetectionT>& src, std::vector<DetectionT>* dst) {
  CHECK(dst);

  std::vector<DetectionT> strong_conf;
  strong_conf.reserve(src.size());
  for (int i = 0; i < src.size(); ++i) {
    if (src[i].conf() > threshold_)
      strong_conf.push_back(src[i]);
  }

  dst->clear();
  nms_->DoNMS(strong_conf, dst);
}

// inline functions
template <typename DetectionT, typename ConfT>
ThresholdAndNMS<DetectionT, ConfT>::ThresholdAndNMS(
    ConfT threshold, std::shared_ptr<NMS<DetectionT> >& nms) 
  : threshold_(threshold), nms_(nms) {

}

template <typename DetectionT, typename ConfT>
ThresholdAndNMS<DetectionT, ConfT>::ThresholdAndNMS(
    ConfT threshold, NMS<DetectionT>* nms)
  : threshold_(threshold), nms_(nms) {

}


template <typename DetectionT, typename ConfT>
inline void ThresholdAndNMS<DetectionT, ConfT>::set_thrshold(
    ConfT threshold) {
  threshold_ = threshold;
}

template <typename DetectionT, typename ConfT>
void ThresholdAndNMS<DetectionT, ConfT>::set_nms(
    std::shared_ptr<NMS<DetectionT> >& nms) {
  nms_ = nms;
}

template <typename DetectionT, typename ConfT>
void ThresholdAndNMS<DetectionT, ConfT>::set_nms(NMS<DetectionT>* nms) {
  nms_ = nms;
}

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_THRESHOLD_AND_NMS_HPP_
