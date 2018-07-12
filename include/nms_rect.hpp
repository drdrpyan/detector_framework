#ifndef BGM_DETECTION_FRAMEWORK_NMS_RECT_HPP_
#define BGM_DETECTION_FRAMEWORK_NMS_RECT_HPP_

#include "nms.hpp"

#include "detection_type.hpp"

#include <glog/logging.h>

#include <list>
#include <vector>

namespace bgm
{

template <typename ConfT, typename CoordT>
class NMSRect : public NMS<DetectionRect<ConfT, CoordT> >
{
  typedef DetectionRect<ConfT, CoordT> DRect;

 public:
  NMSRect(float overlap_threshold = 0.5f);

  virtual void DoNMS(const std::vector<DRect>& in,
                     std::vector<DRect>* out) override;

 private:
  void YMaxArgSort(const std::vector<DRect>& d_rect,
                   std::list<int>* idx) const;
  void CalcArea(const std::vector<DRect>& d_rect,
                  std::vector<float>* area) const;

  float overlap_threshold_;
};

// template fucntions
template <typename ConfT, typename CoordT>
NMSRect<ConfT, CoordT>::NMSRect(float overlap_threshold)
  : overlap_threshold_(overlap_threshold) {

}


template <typename ConfT, typename CoordT>
void NMSRect<ConfT, CoordT>::DoNMS(const std::vector<DRect>& in,
                                   std::vector<DRect>* out) {
  CHECK(result);

  std::list<int> idx;
  YMaxArgSort(in, &idx);

  std::vector<float> area;
  CalcArea(in, &area);

  std::vector<int> pick_idx;
  while (idx.size() > 0) {
    int i = idx.back();
    pick_idx.push_back(i);
    idx.pop_back();

    const cv::Rect_<CoordT>& box1 = in[i].rect();

    auto idx_iter = idx.begin();
    while (idx_iter != idx.end()) {
      const cv::Rect_<CoordT>& box2 = in[*idx_iter].rect();
      CoordT x_min = std::max(box1.x, box2.x);
      CoordT y_min = std::max(box1.y, box2.y);
      CoordT x_max = std::min(box1.x + box1.width, box2.x + box2.width);
      CoordT y_max = std::min(box1.y + box1.height, box2.y + box2.height);
      
      CoordT bound_area = 
          std::max(x_max - x_min, static_cast<CoordT>(0)) * 
            std::max(y_max - y_min, static_cast<CoordT>(0));
      float overlap = bound_area / area[*idx_iter];

      if (overlap > overlap_threshold_)
        idx.erase(idx_iter++);
      else
        ++idx_iter;
    }
  }

  out->resize(pick_idx.size());
  for (int i = 0; i < pick_idx.size(); ++i)
    (*out)[i] = in[pick_idx[i]];
}

template <typename ConfT, typename CoordT>
void NMSRect<ConfT, CoordT>::YMaxArgSort(
    const std::vector<DRect>& d_rect, std::list<int>* idx) const {
  CHECK(idx);

  idx->resize(d_rect.size());
  std::vector<int> idx_vec(d_rect.size());
  std::iota(idx_vec.begin(), idx_vec.end(), 0);

  std::sort(idx_vec.begin(), idx_vec.end(),
            [&detection](int i1, int i2) {
                return (d_rect[i1].rect().y + d_rect[i1].rect().height) < (d_rect[i2].rect().y + d_rect[i2].rect().height); });

  idx->assign(idx_vec.begin(), idx_vec.end());
}

template <typename ConfT, typename CoordT>
inline void NMSRect<ConfT, CoordT>::CalcArea(
    const std::vector<DRect>& d_rect, std::vector<float>* area) const {
  CHECK(area);

  area->resize(d_rect.size());
  for (int i = 0; i < d_rect.size(); ++i)
    (*area)[i] = d_rect[i].rect().area();
}

} // namespace bgm
#endif // !BGM_DETECTION_FRAMEWORK_NMS_RECT_HPP_