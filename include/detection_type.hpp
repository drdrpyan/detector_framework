#ifndef BGM_DETECTION_FRAMEWORK_TYPE_HPP_
#define BGM_DETECTION_FRAMEWORK_TYPE_HPP_

#include <opencv2/core.hpp>

#include <vector>

namespace bgm
{

template <typename ConfT>
class DetectionObj
{
 public:
  DetectionObj();
  template <typename InIterT>
  DetectionObj(ConfT conf,
               const InIterT& label_conf_beg,
               const InIterT& label_conf_end);
  ConfT conf() const;
  int max_class_idx() const;
  const std::vector<ConfT> class_conf() const;
  ConfT class_conf(int label_idx) const;
  void set_conf(ConfT conf);
  template <typename InIterT>
  void set_label_idx(const InIterT& label_conf_beg,
                     const InIterT& label_conf_end);

 private:
  ConfT conf_;
  int max_class_idx_;
  std::vector<ConfT> class_conf_;
};

template <typename ConfT, typename CoordT>
class DetectionRect : public DetectionObj<ConfT>
{
 public:
  DetectionRect();
  template <typename InIterT>
  DetectionRect(ConfT conf,
                const InIterT& label_conf_beg,
                const InIterT& label_conf_end,
                const cv::Rect_<CoordT>& rect);
  const cv::Rect_<CoordT>& rect() const;
  void set_rect(const cv::Rect_<CoordT>& rect);
 private:
  cv::Rect_<CoordT> rect_;
};


//template <typename CoordT>
//class Rect
//{
// public:
//  bool center_origin() const;
//  void TransformOrigin(bool center_origin);
//  Rect<CoordT> ToCenterOriginRect();
//  Rect<CoordT> ToTopLeftOriginRect();
//
//  CoordT x;
//  CoordT y;
//  CoordT width;
//  CoordT height;
//
// private:
//  bool center_origin_;
//};
//
//template <typename ConfT, typename CoordT>
//class DetectionRect : public DetectionObj<ConfT>
//{
// public:
//  const Rect<CoordT>& rect() const;
//  void set_rect(const Rect<CoordT>& rect);
//
// private:
//  Rect<CoordT> rect_;
//};
} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_TYPE_HPP_

