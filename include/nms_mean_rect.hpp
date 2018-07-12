#ifndef BGM_DETECTION_FRAMEWORK_NMS_MEAN_RECT_HPP_
#define BGM_DETECTION_FRAMEWORK_NMS_MEAN_RECT_HPP_

#include "nms_rect.hpp"

namespace bgm
{

template <typename ConfT, typename CoordT>
class NMSMeanRect : public NMSRect<ConfT, CoordT>
{
  typedef DetectionRect<ConfT, CoordT> DRect;

 public:
  NMSMeanRect(float overlap_threshold 0.5f);
  virtual void DoNMS(const std::vector<DRect>& in,
                     std::vector<DRect>* out) override;
};

// template functions
template <typename ConfT, typename CoordT>
void NMSMeanRect<ConfT, CoordT>::DoNMS(const std::vector<DRect>& in,
                                       std::vector<DRect>* out) {
  CHECK(out);
  out->clear();

  std::list<int> idx;
  ConfMaxArgSort(in, &idx);

  std::vector<float> area;
  CalcArea(in, &area);

  std::vector<int> pick_idx;
  while (idx.size() > 0) {
    int i = idx.back();
    pick_idx.push_back(i);
    idx.pop_back();

    const cv::Rect_<CoordT>& box1 = in[i].rect();

    cv::Rect_<CoordT> avg_box(box1);
    int overlap_cnt = 1;

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

      if (overlap > overlap_threshold()) {
        idx.erase(idx_iter++);

        avg_box.x += box2.x;
        avg_box.y += box2.y;
        avg_box.width += box2.width;
        avg_box.height += box2.height;
        ++overlap_cnt;
      }
      else
        ++idx_iter;
    }

    avg_box.x /= overlap_cnt;
    avg_box.y /= overlap_cnt;
    avg_box.width /= overlap_cnt;
    avg_box.height /= overlap_cnt;

    DRect mean_nms_rect = in[i].rect();
    mean_nms_rect.set_rect(avg_box);
    out->push_back(mean_nms_rect);
  }
}

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_NMS_MEAN_RECT_HPP_

