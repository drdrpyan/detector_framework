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
  typedef DetectionRect<Dtype, Dtype> CaffeDetectionT;

 public:
  YOLOV2CaffeDetector();
  YOLOV2CaffeDetector(CaffeWrapper<Dtype>* caffe,
                      YOLOv2CaffeDecoder<Dtype>* decoder,
                      DetectionFilter<CaffeDetectionT>* filter = nullptr,
                      bool do_filtering = false);
  void set_caffe(CaffeWrapper<Dtype>* caffe);
  void set_caffe_decoder(YOLOv2CaffeDecoder<Dtype>* decoder);

 protected:
  virtual void Detect_impl(const cv::Mat& img, 
                           std::vector<CaffeDetectionT>* result) override;
  virtual void Detect_impl(
      const std::vector<cv::Mat>& imgs,
      std::vector<std::vector<CaffeDetectionT> >* result) override;

 private:
  std::shared_ptr<CaffeWrapper<Dtype> > caffe_;
  std::shared_ptr<YOLOv2CaffeDecoder<Dtype> > caffe_decoder_;
}; // class YOLOV2CaffeDetector


// template fuctions
template <typename Dtype>
inline YOLOV2CaffeDetector<Dtype>::YOLOV2CaffeDetector() {

}

template <typename Dtype>
inline YOLOV2CaffeDetector<Dtype>::YOLOV2CaffeDetector(
    CaffeWrapper<Dtype>* caffe, YOLOv2CaffeDecoder<Dtype>* decoder,
    DetectionFilter<CaffeDetectionT>* filter, bool do_filtering) 
  : Detector<CaffeDetectionT>(filter, do_filtering), 
    caffe_(caffe), decoder_(decoder) {

}

template <typename Dtype>
inline void YOLOV2CaffeDetector<Dtype>::set_caffe(
    CaffeWrapper<Dtype>* caffe) {
  caffe_.reset(caffe);
}

template <typename Dtype>
inline void YOLOV2CaffeDetector<Dtype>::set_caffe_decoder(
    YOLOv2CaffeDecoder<Dtype>* decoder) {
  caffe_decoder_.reset(decoder);
}

template <typename Dtype>
inline void YOLOV2CaffeDetector<Dtype>::Detect_impl(
    const cv::Mat& img, std::vector<CaffeDetectionT>* result) {
  caffe_->set_input(0, img);
  caffe_->Process();
  caffe_decoder_->Decode(caffe_->output(0), result);
}

template <typename Dtype>
void YOLOV2CaffeDetector<Dtype>::Detect_impl(
    const std::vector<cv::Mat>& imgs,
    std::vector<std::vector<CaffeDetectionT> >* result) {
  caffe_->set_input(0, imgs);
  caffe_->Process();
  caffe_decoder_->Decode(caffe_->output(0), result);
}


} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DETECTOR_HPP_
