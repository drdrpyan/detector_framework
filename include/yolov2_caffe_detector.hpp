#ifndef BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DETECTOR_HPP_
#define BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DETECTOR_HPP_

#include "detector.hpp"

#include "detection_type.hpp"
#include "caffe_wrapper.hpp"
#include "yolov2_caffe_decoder.hpp"

#include <memory>

namespace bgm
{

template <typename Dtype>
class YOLOV2CaffeDetector : public Detector<DetectionRect<Dtype, Dtype> >
{
 public:
 protected:
  template <typename OutIterT>
  virtual void Detect_impl(const cv::Mat& img, OutIterT& out_beg);
  template <typename InIterT, typename OutIterT>
  virtual void Detect_impl(const InIterT& img_beg, const InIterT& img_end,
                           OutIterT& out_beg);

 private:
  std::shared_ptr<CaffeWrapper<Dtype> > caffe_;
  std::shared_ptr<YOLOv2CaffeDecoder<Dtype> > caffe_decoder_;
}; // class YOLOV2CaffeDetector


// template fuctions
template <typename Dtype>
template <typename OutIterT>
inline void YOLOV2CaffeDetector<Dtype>::Detect_impl(
    const cv::Mat& img, OutIterT& out_beg) {
  caffe_->set_input(0, img);
  caffe_->Process();
  caffe_decoder_->Decode(caffe_->output(0), out_beg);
}

template <typename InIterT, typename OutIterT>
void YOLOV2CaffeDetector<Dtype>::Detect_impl(
    const InIterT& img_beg, const InIterT& img_end, OutIterT& out_beg) {

}


} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DETECTOR_HPP_
