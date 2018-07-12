#ifndef BGM_DETECTION_FRAMEWORK_DETECTION_FILTER_HPP_
#define BGM_DETECTION_FRAMEWORK_DETECTION_FILTER_HPP_

#include <vector>

namespace bgm
{

template <typename DetectionT>
class DetectionFilter
{
 public:
  virtual void Filter(const std::vector<DetectionT>& src,
                      std::vector<DetectionT>* dst) = 0;
}; // class DetectionFilter
} // namespace bgm
#endif // !BGM_DETECTION_FRAMEWORK_DETECTION_FILTER_HPP_
