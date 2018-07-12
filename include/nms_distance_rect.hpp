#ifndef BGM_DETECTION_FRAMEWORK_NMS_DISTANCE_RECT_HPP_
#define BGM_DETECTION_FRAMEWORK_NMS_DISTANCE_RECT_HPP_

#include "nms_rect.hpp"

#include <memory>

namespace bgm
{

template <typename ConfT, typename CoordT>
class NMSDistanceRect : public NMSRect<ConfT, CoordT>
{
 public:
  enum DistanceUnit {WIDTH, HEIGHT, DIAGONAL};
 private:
  typedef DetectionRect<ConfT, CoordT> DRect;

 public:
  NMSDistanceRect(NMSRect<ConfT, CoordT>* base_nms,
                  float dist_mult, DistanceUnit unit = HEIGHT);
  virtual void DoNMS(const std::vector<DRect>& in,
                     std::vector<DRect>* out) override;

 private:
  bool IsNear(const cv::Rect_<CoordT>& rect1,
              const cv::Rect_<CoordT>& rect2) const;
  float Distance(const cv::Rect_<CoordT>& rect1,
                 const cv::Rect_<CoordT>& rect2) const;

  std::shared_ptr<NMSRect<ConfT, CoordT> > base_nms_;
  float dist_mult_;
  DistanceUnit distance_unit_;
}; // class NMSDistanceRect

// template functions
template <typename ConfT, typename CoordT>
NMSDistanceRect<ConfT, CoordT>::NMSDistanceRect(
    NMSRect<ConfT, CoordT>* base_nms, float dist_mult, DistanceUnit unit)
  : base_nms_(base_nms), dist_mult_(dist_mult), distance_unit_(unit) {
  CHECK(base_nms);
  CHECK_GT(dist_mult, 0);
}

template <typename ConfT, typename CoordT>
void NMSDistanceRect<ConfT, CoordT>::DoNMS(const std::vector<DRect>& in,
                                           std::vector<DRect>* out) {
  out->clear();

  std::vector<DRect> base_result;
  base_nms_->DoNMS(in, &base_result);

  std::list<DRect> temp_result(base_result.begin(), 
                               base_result.end());
  auto iter1 = temp_result.begin();
  while (iter1 != temp_result.end()) {
    auto iter2 = iter1;
    ++iter2;
    while (iter2 != temp_result.end()) {
      if (IsNear(iter1->rect(), iter2->rect()))
        temp_result.erase(iter2++);
      else
        ++iter2;
    }

    ++iter1;
  }

  out->assign(temp_result.begin(), temp_result.end());
}

template <typename ConfT, typename CoordT>
bool NMSDistanceRect<ConfT, CoordT>::IsNear(
    const cv::Rect_<CoordT>& rect1, const cv::Rect_<CoordT>& rect2) const {
  float dist = Distance(rect1, rect2);
  float limit;
  switch (distance_unit_) {
    case HEIGHT:
      limit = rect1.height * dist_mult_;
      break;
    case DIAGONAL:
      float diag = std::sqrtf(std::powf(rect1.width, 2), 
                              std::powf(rect1.height, 2));
      limit = diag * dist_mult_;
      break;
    case WIDTH:
      limit = rect1.width * dist_mult_;
      break;
    default:
      LOG(FATAL) << "Undefined distance unit";
  }
  return (dist < limit);
}

template <typename ConfT, typename CoordT>
inline float NMSDistanceRect<ConfT, CoordT>::Distance(
    const cv::Rect_<CoordT>& rect1, const cv::Rect_<CoordT>& rect2) const {
  return std::sqrtf(std::powf(rect1.x - rect2.x, 2) + std::powf(rect1.y - rect2.y, 2));
}
} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_NMS_DISTANCE_RECT_HPP_
