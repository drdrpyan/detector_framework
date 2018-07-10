#ifndef BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DECODER_HPP_
#define BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DECODER_HPP_

#include "detection_type.hpp"

#include <opencv2/core.hpp>

#include <glog/logging.h>
#include <caffe/blob.hpp>

namespace bgm
{

template <typename Dtype>
class YOLOv2CaffeDecoder
{
 public:
  YOLOv2CaffeDecoder();
  template <typename OutIterT>
  void Decode(const caffe::Blob<Dtype>& yolov2_out,
              OutIterT& out_begin) const;


  //void set_conf_idx(int conf_idx);
  //void set_label_idx_begin(int label_idx_begin);
  void SetDecodeOrder(int bbox_conf, int label_conf, int bbox);

  void set_num_class(int num);
  template <typename InIterT>
  void set_anchor_box(const InIterT& anchor_begin, 
                      const InIterT& anchor_end);
  template <typename InIterT>
  void set_anchor_weight(const InIterT& weight_begin,
                         const InIterT& weight_end);
  void set_grid_cell_size(const cv::Size2f& grid_cell_size);
  void set_grid_cell_size(float width, float height);
  template <typename InIterT>
  void set_batch_offset(const InIterT& offset_begin,
                        const InIterT& offset_end); 

 private:
  //bool VerifyParams() const;
  //void DecodeBBoxConf(int anchor_idx, int ch_step, 
  //                    Dtype*& iter, DetectionRect<Dtype, Dtype>& detected_obj) const;
  //void DecodeLabelConf(int anchor_idx, int ch_step, 
  //                     Dtype*& iter, DetectionRect<Dtype, Dtype>& detected_obj) const;
  //void DecodeBBox(int anchor_idx, int ch_step, 
  //                Dtype*& iter, DetectionRect<Dtype, Dtype>& detected_obj) const;

  Dtype DecodeConf(const Dtype* batch_begin, int anchor_idx,
                   int ch_step);
  cv::Rect_<Dtype> DecodeBBox(const Dtype* batch_begin, int anchor_idx,
                              int ch_step, int cell_x, int cell_y);
  template <typename OutIterT>
  void DecodeLabelConf(const Dtype* batch_begin, int anchor_idx,
                       int ch_step, OutIterT& out_beg);

  cv::Rect_<Dtype> YOLOBOXToAbsBox(Dtype yolo_x, Dtype yolo_y,
                                   Dtype yolo_w, Dtype yolo_h,
                                   Dtype offset_x = 0, 
                                   Dtype offset_y = 0);
  int ComputeAnchorElemChannel(int anchor_idx, int elem_ch) const;
  Dtype Sigmoid(Dtype value) const
  

  ////template <typename Dtype>
  //void (*decode_function[3])(int anchor_idx, int ch_step,
  //                           Dtype*& iter, 
  //                           DetectionRect<Dtype, Dtype>& detected_obj);

  int num_class_;
  std::vector<cv::Rect_<Dtype> > anchor_box_;
  std::vector<float> anchor_weight_;
  cv::Size2f grid_cell_size_;
  std::vector<cv::Point2f> batch_offset_;
  int conf_ch_;
  int bbox_ch_begin_;
  int class_ch_begin_;
};

// inline functions
template <typename Dtype>
inline void YOLOv2CaffeDecoder<Dtype>::set_num_class(int num) {
  CHECK_GE(num, 0) << "# class is not positive";

  num_class_ = num;
}

template <typename Dtype>
template <typename InIterT>
inline void YOLOv2CaffeDecoder<Dtype>::set_anchor_box(
    const InIterT& anchor_begin, const InIterT& anchor_end) {
  anchor_box_.assign(anchor_begin, anchor_end);
}

template <typename Dtype>
template <typename InIterT>
inline void YOLOv2CaffeDecoder<Dtype>::set_anchor_weight(
    const InIterT& weight_begin, const InIterT& weight_end) {
  anchor_weight_.assign(weight_begin, weight_end);
}

template <typename Dtype>
inline void YOLOv2CaffeDecoder<Dtype>::set_grid_cell_size(
    const cv::Size2f& grid_cell_size) {
  grid_cell_size_ = grid_cell_size;
}

template <typename Dtype>
inline void YOLOv2CaffeDecoder<Dtype>::set_grid_cell_size(
    float width, float height) {
  CHECK_GT(width, 0) << "width must be positive";
  CHECK_GT(height, 0) << "height must be positive";
  grid_cell_size_.width = width;
  grid_cell_size_.height = height;
}

template <typename Dtype>
template <typename InIterT>
inline void YOLOv2CaffeDecoder<Dtype>::set_batch_offset(
    const InIterT& offset_begin, const InIterT& offset_end) {
  batch_offset_.assign(offset_begin, offset_end);
}

// template functions
//template <typename Dtype>
//bool YOLOv2CaffeDecoder<Dtype>::VerifyParams() const {
//  
//}

template <typename Dtype>
void YOLOv2CaffeDecoder<Dtype>::SetDecodeOrder(int bbox_conf,
                                       int label_conf,
                                       int bbox) {
  CHECK_GE(bbox_conf, 0) << "order must be 0 <= order < 3";
  CHECK_LT(bbox_conf, 3) << "order must be 0 <= order < 3";
  CHECK_GE(label_conf, 0) << "order must be 0 <= order < 3";
  CHECK_LT(label_conf, 3) << "order must be 0 <= order < 3";
  CHECK_GE(bbox, 0) << "order must be 0 <= order < 3";
  CHECK_LT(bbox, 3) << "order must be 0 <= order < 3";
  CHECK_NE(bbox_conf, label_conf) 
    << "orders must be different from each other";
  CHECK_NE(label_conf, bbox)
    << "orders must be different from each other";
  CHECK_NE(bbox_conf, bbox)
    << "orders must be different from each other";

  decode_function[bbox_conf] = DecodeBBoxConf;
  decode_function[label_conf] = DecodeLabelConf;
  decode_function[bbox] = DecodeBBox;
}

template <typename Dtype>
template <typename OutIterT>
void YOLOv2CaffeDecoder<Dtype>::Decode(const caffe::Blob<Dtype>& yolov2_out,
                               OutIterT& out_begin) const {
  CHECK_EQ(anchor_box_.size(), anchor_weight_())
    << "(# of anchor box) is not equal to (# of anchor box weight)";
  CHECK_EQ((5 + num_class_) * anchor_box_.size(), yolov2_out.channels())
    << "channel of YOLOv2 output is not (5+#class)*(# anchor box)";
  
  const int IN_WIDTH = yolov2_out.width();
  const int IN_HEIGHT = yolov2_out.height();
  const int CH_STEP = IN_WIDTH * IN_HEIGHT;
  const int ANCHOR_STEP = CH_STEP * (5 + num_class_);

  // 여기서부터 다시 할 것
  const Dtype* anchor_iter = yolov2_out.cpu_data();
  for (int i = 0; i < anchor_box_.size(); ++i) {
    const Dtype* cell_iter = anchor_iter;

    for (int j = 0; j < IN_WIDTH; ++j) {
      for (int k = 0; k < IN_HEIGHT; ++k) {
        const Dtype* ch_iter = cell_iter++;

        DetectionRect<Dtype, Dtype> detection_rect;
        for (int l = 0; l < 3; ++l)
          decode_function[l](CH_STEP, i, ch_iter, detection_rect);
        *out_begin++ = detection_rect;
      }
    }

    anchor_iter += ANCHOR_STEP;
  }
}

//template <typename Dtype>
//void YOLOv2CaffeDecoder<Dtype>::DecodeBBoxConf(
//    int anchor_idx, int ch_step, Dtype*& iter, 
//    DetectionRect<Dtype, Dtype>& detected_obj) const {
//  Dtype yolo_x = *iter;
//  iter += ch_step;
//  Dtype yolo_y = *iter;
//  iter += ch_step;
//  Dtype yolo_w = *iter;
//  iter += ch_step;
//  Dtype yolo_h = *iter;
//
//
//}

template <typename Dtype>
Dtype YOLOv2CaffeDecoder<Dtype>::DecodeConf(
    const Dtype* batch_begin, int anchor_idx, int ch_step) {
  CHECK(batch_begin);
  CHECK_GE(anchor_idx, 0);
  CHECK_LT(anchor_idx, anchor_box_.size());
  CHECK_GT(ch_step, 0);

  int ch = ComputeAnchorElemChannel(anchor_idx, conf_ch_);
  Dtype conf = *(batch_begin + (ch_step*ch));
  return Sigmoid(conf);
}

template <typename Dtype>
cv::Rect_<Dtype> YOLOv2CaffeDecoder<Dtype>::DecodeBBox(
    const Dtype* batch_begin, int anchor_idx,
    int ch_step, int cell_x, int cell_y) {
  CHECK(batch_begin);
  CHECK_GE(anchor_idx, 0);
  CHECK_LT(anchor_idx, anchor_box_.size());
  CHECK_GT(ch_step, 0);
  CHECK_GE(cell_x, 0);
  CHECK_GE(cell_y, 0);

  int ch = ComputeAnchorElemChannel(anchor_idx, bbox_ch_begin_);
  Dtype* iter = batch_begin + (ch_step*ch);

  Dtype yolo_x = *iter;
  iter += ch_step;
  Dtype yolo_y = *iter;
  iter += ch_step;
  Dtype yolo_w = *iter;
  iter += ch_step;
  Dtype yolo_h = *iter;

  cv::Rect_<Dtype>& anchor = anchor_box_[anchor_idx];

  cv::Rect_<Dtype> bbox;
  bbox.width = std::exp(yolo_w) * anchor.width;
  bbox.height = std::exp(yolo_h) * anchor.height;
  raw_box.x = (Sigmoid(yolo_x) * anchor.width) - (bbox.width / 2.);
  raw_box.y = (Sigmoid(yolo_y) * anchor.height) - (bbox.height / 2.);

  raw_box.x += grid_cell_size_.width * cell_x;
  raw_box.x += grid_cell_size_.height * cell_y;

  return bbox;
}

template <typename Dtype>
template <typename OutIterT>
void YOLOv2CaffeDecoder<Dtype>::DecodeLabelConf(
    const Dtype* batch_begin, int anchor_idx,
    int ch_step, OutIterT& out_beg) {
  CHECK(batch_begin);
  CHECK_GE(anchor_idx, 0);
  CHECK_LT(anchor_idx, anchor_box_.size());
  CHECK_GT(ch_step, 0);

  int ch = ComputeAnchorElemChannel(anchor_idx, class_ch_begin_);
  Dtype* iter = batch_begin + (ch_step*ch);
  auto out_iter = out_beg;
  for (int i = 0; i < num_class_; ++i) {
    *out_iter++ = *iter;
    iter += ch_step;
  }
}

template <typename Dtype>
int YOLOv2CaffeDecoder<Dtype>::ComputeAnchorElemChannel(
    int anchor_idx, int elem_ch) const {
  CHECK_GE(anchor_idx, 0);
  CHECK_LT(anchor_idx, anchor_box_.size());
  CHECK_GE(elem_ch, 0);
  CHECK_LT(elem_ch, 5 + num_class_);

  return ((5 + num_class_)*anchor_idx) + elem_ch;
}

template <typename Dtype>
inline Dtype YOLOv2CaffeDecoder<Dtype>::Sigmoid(Dtype value) const {
  return 1. / (1. + std::exp(-value));
}

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_YOLOV2_CAFFE_DECODER_HPP_