#ifndef BGM_DETECTION_FRAMEWORK_NMS_HPP_
#define BGM_DETECTION_FRAMEWORK_NMS_HPP_

#include <vector>

namespace bgm
{

template <typename DetectionT>
class NMS
{
 public:
  virtual void DoNMS(const std::vector<DetectionT>& in,
                     std::vector<DetectionT>* out) = 0;
};

} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_NMS_HPP_
