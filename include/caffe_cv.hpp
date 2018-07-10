#ifndef BGM_DETECTION_FRAMEWORK_CAFFE_CV_HPP_
#define BGM_DETECTION_FRAMEWORK_CAFFE_CV_HPP_

#include <opencv2/core.hpp>

#include <caffe/blob.hpp>

namespace bgm
{

template <typename Dtype>
class CaffeCV
{
 public:
  static void CVMatToBlob(const cv::Mat& in_mat, 
                          caffe::Blob<Dtype>* out_blob);

  template <typename InIterT>
  static void CVMatToBlob(const InIterT& in_mat_beg, 
                          const InIterT& in_mat_end,
                          caffe::Blob<Dtype>* out_blob);

  static cv::Mat BlobToCVMat(const caffe::Blob<Dtype>& blob,
                             int batch_idx = 0);

  template <typename OutIterT>
  static void BlobToCVMat(const caffe::Blob<Dtype>& blob,
                          OutIterT& out_mat_beg);


 private:
  static cv::Mat ConvertCVMatDepth(const cv::Mat& mat);
  static void ReshapeBlobLikeCVMat(const cv::Mat& mat, 
                                   caffe::Blob<Dtype>& blob);
  template <typename InIterT>
  static void ReshapeBlobLikeCVMat(const InIterT& mat_beg,
                                   const InIterT& mat_end,
                                   caffe::Blob<Dtype>& blob);
  static void CVMatToBlobData(const cv::Mat& in_mat,
                              Dtype* dst);
  static cv::Mat CreateCVMat(const cv::Size& size, int ch,
                             Dtype* data);
}; // class CaffeCV

// static template fucntions
template <typename Dtype>
void CaffeCV<Dtype>::CVMatToBlob(const cv::Mat& in_mat,
                                 caffe::Blob<Dtype>* out_blob) {
  CHECK(out_blob);

  ReshapeBlobLikeCVMat(in_mat, *out_blob);
  CVMatToBlobData(in_mat, out_blob->mutable_cpu_data());
}

template <typename Dtype>
template <typename InIterT>
void CaffeCV<Dtype>::CVMatToBlob(const InIterT& in_mat_beg,
                                 const InIterT& in_mat_end,
                                 caffe::Blob<Dtype>* out_blob) {
  CHECK(out_blob);
  CHECK_GT(std::distance(in_mat_beg, in_mat_end), 0);

  ReshapeBlobLikeCVMat(in_mat_beg, in_mat_end, *out_blob);
  Dtype* out_blob_iter = out_blob->mutable_cpu_data();
  int batch_step = out_blob->count(1);
  for (auto iter = in_mat_beg; iter != in_mat_end; ++iter) {
    CVMatToBlobData(*iter, out_blob_iter);
    out_blob_iter += batch_step;
  }
}

template <typename Dtype>
cv::Mat CaffeCV<Dtype>::BlobToCVMat(const caffe::Blob<Dtype>& blob,
                                    int batch_idx) {
  CHECK_GE(batch_idx, 0);
  CHECK_LT(batch_idx, blob.num());
  CHECK_LE(blob.channels(), 4);

  const Dtype* src_data = blob.cpu_data() + blob.offset(batch_idx);
  int src_length = blob.count(1);
  std::vector<Dtype> buffer(src_length);
  std::copy(src_data, src_data + src_length, &(buffer[0]));

  cv::Size mat_size(blob.width(), blob.height());
  Dtype* buffer_iter = &(buffer[0]);
  int step = mat_size.area();
  std::vector<cv::Mat> mat_channels(blob.channels());
  for (int i = 0; i < mat_channels.size(); ++i) {
    mat_channels[i] = CreateCVMat(size, 1, buffer_iter);
    buffer_iter += step;
  }

  cv::Mat merged;
  cv::merge(mat_channels, merged);
  return merged;
}

template <typename Dtype>
template <typename OutIterT>
inline void CaffeCV<Dtype>::BlobToCVMat(const caffe::Blob<Dtype>& blob,
                                        OutIterT& out_mat_beg) {
  for (int i = 0; i < blob.num(); ++i)
    *out_mat_beg++ = BlobToCVMat(blob, i);
}

template <typename Dtype>
cv::Mat CaffeCV<Dtype>::ConvertCVMatDepth(const cv::Mat& mat) {
  LOG(FATAL) << "ConvertCVMatDepth<" 
    << typeid(Dtype).name() << "> is not implemented";
}

template <>
cv::Mat CaffeCV<float>::ConvertCVMatDepth(const cv::Mat& mat) {
  if (mat.depth() != CV_32F) {
    cv::Mat converted;
    mat.convertTo(converted, CV_MAKETYPE(CV_32F, mat.channels()));
    return converted;
  }
  else
    return mat;
}

template <>
cv::Mat CaffeCV<double>::ConvertCVMatDepth(const cv::Mat& mat) {
  if (mat.depth() != CV_64F) {
    cv::Mat converted;
    mat.convertTo(converted, CV_MAKETYPE(CV_64F, mat.channels()));
    return converted;
  }
  else
    return mat;
}

template <typename Dtype>
void CaffeCV<Dtype>::ReshapeBlobLikeCVMat(const cv::Mat& mat,
                                          caffe::Blob<Dtype>& blob) {
  std::vector<int> shape(4, 1);
  shape[1] = mat.channels();
  shape[2] = mat.rows;
  shape[3] = mat.cols;
  blob.Reshape(shape);
}

template <typename Dtype>
template <typename InIterT>
void CaffeCV<Dtype>::ReshapeBlobLikeCVMat(const InIterT& mat_beg,
                                          const InIterT& mat_end,
                                          caffe::Blob<Dtype>& blob) {
  const int NUM_MAT = std::distance(mat_beg, mat_end);
  CHECK_GT(NUM_MAT, 0);
  
  std::vector shape(4);
  shape[0] = NUM_MAT;
  shape[1] = mat_beg->channels();
  shape[2] = mat_beg->rows;
  shape[3] = mat_beg->cols;

  for (auto iter = mat_beg + 1; iter != mat_end; ++iter) {
    CHECK_EQ(iter->channels(0), shape[1]);
    CHECK_EQ(iter->rows, shape[2]);
    CHECK_EQ(iter->cols, shape[3]);
  }

  blob.Reshape(shape);
}

template <typename Dtype>
void CaffeCV<Dtype>::CVMatToBlobData(const cv::Mat& in_mat,
                                     Dtype* dst) {
  cv::Mat depth_converted = ConvertCVMatDepth(in_mat);
  cv::Size size(depth_converted.cols, depth_converted.rows);
  Dtype* dst_iter = dst;
  int step = size.area();

  std::vector<cv::Mat> mat_channels(depth_converted.channels());
  for (int i = 0; i < mat_channels.size(); ++i) {
    mat_channels[i] = CreateCVMat(size, 1, dst_iter);
    dst_iter += step;
  }

  cv::split(depth_converted, mat_channels);
}

template <typename Dtype>
inline cv::Mat CaffeCV<Dtype>::CreateCVMat(const cv::Size& size, int ch,
                                           Dtype* data) {
  LOG(FATAL) << "CreateCVMat<" 
    << typeid(Dtype).name() << "> is not implemented";
}

template <>
inline cv::Mat CaffeCV<float>::CreateCVMat(const cv::Size& size, int ch,
                                           float* data) {
  CHECK_GT(ch, 0);
  CHECK(data);
  return cv::Mat(size, CV_MAKETYPE(CV_32F, ch), data);
}

template <>
inline cv::Mat CaffeCV<double>::CreateCVMat(const cv::Size& size, int ch,
                                            double* data) {
  CHECK_GT(ch, 0);
  CHECK(data);
  return cv::Mat(size, CV_MAKETYPE(CV_32F, ch), data);
}

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_CAFFE_CV_HPP_
