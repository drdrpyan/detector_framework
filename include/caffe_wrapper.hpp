#ifndef TLR_DETECTION_FRAMEWORK_CAFFE_WRAPPER_HPP_
#define TLR_DETECTION_FRAMEWORK_CAFFE_WRAPPER_HPP_

#include "caffe_cv.hpp"

#include "caffe/caffe.hpp"

#include <memory>
#include <typeinfo>

namespace bgm
{

template <typename Dtype>
class CaffeWrapper
{
 public:
  CaffeWrapper(const std::string& net_file,
               const std::string& model_file,
               bool use_gpu = true);
  virtual void Process();

  void set_input(int idx, const caffe::Blob<Dtype>& blob, bool copy = false);
  void set_input(int idx, const cv::Mat& mat);
  void set_input(int idx, const std::vector<cv::Mat>& mat);

  caffe::Blob<Dtype>& input(int idx) const;
  caffe::Blob<Dtype>& output(int idx) const;

 private:
  //void set_input_impl(int idx, const cv::Mat& mat, int mat_depth);

  std::shared_ptr<caffe::Net<float> > net_;
};

typedef CaffeWrapper<float> CaffeSingle;
typedef CaffeWrapper<double> CaffeDouble;

// template functions
template <typename Dtype>
CaffeWrapper<Dtype>::CaffeWrapper(const std::string& net_file,
                                  const std::string& model_file,
                                  bool use_gpu) {
    if (use_gpu) {
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  }
  else 
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

  net_.reset(new caffe::Net<float>(net_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(model_file);

  input_.resize(net_->num_inputs());
  output_.resize(net_->num_outputs());
}

template <typename Dtype>
inline void CaffeWrapper<Dtype>::Process() {
  net_->Forward();
}


template <typename Dtype>
void CaffeWrapper<Dtype>::set_input(int idx, const caffe::Blob<Dtype>& blob, bool copy) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, net_->num_inputs());
  
  caffe::Blob<Dtype>* target = net_->input_blobs()[idx];
  if (copy)
    target->CopyFrom(blob, false, true);
  else {
    target->ReshapeLike(blob);
    target->ShareData(blob);
  }
}

//template <>
//inline void CaffeWrapper<float>::set_input(int idx, const cv::Mat& mat) {
//  set_input_impl(idx, mat, CV_32F);
//}
//
//template <>
//void CaffeWrapper<double>::set_input(int idx, const cv::Mat& mat) {
//  set_input_impl(idx, mat, CV_64F);
//}

template <typename Dtype>
inline void CaffeWrapper<Dtype>::set_input(int idx, const cv::Mat& mat) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, net_->num_inputs());
  CaffeCV<Dtype>::CVMatToBlob(mat, net_->input_blobs()[idx]);
  //LOG(FATAL) << "Not implemented for CaffeWrapper<" << typeid(Dtype).name() << ">.";
}

template <typename Dtype>
inline void set_input(int idx, const std::vector<cv::Mat>& mat) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, net_->num_inputs());
  CaffeCV<Dtype>::CVMatToBlob(mat.cbegin(), mat.cend(),
                              net_->input_blobs()[idx]);
}

//template <typename Dtype>
//void CaffeWrapper<Dtype>::set_input_impl(int idx, const cv::Mat& mat, int mat_depth) {
//  CHECK_GE(idx, 0);
//  CHECK_LT(idx, net_->num_inputs());
//
//  cv::Mat converted;
//  if (mat.depth() != mat_depth)
//    mat.convertTo(converted, CV_MAKETYPE(mat_depth, mat.channels()));
//  else
//    converted = mat;
//
//  std::vector<int> mat_shape(4, 1);
//  mat_shape[1] = mat.depth();
//  mat_shape[2] = mat.rows;
//  mat_shape[3] = mat.cols;
//
//  caffe::Blob<float>* target = net_->input_blobs()[idx];
//  target->Reshape(mat_shape);
//  
//  cv::Size mat_size(mat_shape[3], mat_shape[2]);
//  int ch_type = CV_MAKETYPE(mat_depth, 1);
//  float* blob_data = target->mutable_cpu_data();
//  int step = mat_shape[2] * mat_shape[3];  
//  cv::Mat mat_channels[3];
//  mat_channels[0] = cv::Mat(mat_size, ch_type, blob_data);
//  mat_channels[1] = cv::Mat(mat_size, ch_type, blob_data + step);
//  mat_channels[2] = cv::Mat(mat_size, ch_type, blob_data + (step*2));
//  cv::split(converted, mat_channels); 
//}

template <typename Dtype>
inline caffe::Blob<Dtype>& CaffeWrapper<Dtype>::input(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, net_->num_inputs());
  return *(net_->input_blobs()[idx]);
}

template <typename Dtype>
inline caffe::Blob<Dtype>& CaffeWrapper<Dtype>::output(int idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, net_->num_outputs());
  return *(net_->num_outputs()[idx]);
}

} // namespace bgm

#endif // !TLR_CAFFE_WRAPPER_HPP_
