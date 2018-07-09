#ifndef BGM_DETECTION_FRAMEWORK_NMS_HPP_
#define BGM_DETECTION_FRAMEWORK_NMS_HPP_

namespace bgm
{

template <typename DetectionT>
class NMS
{
 public:
  template <typename InIterT, typename OutIterT>
  virtual void nms(const InIterT& in_beg, const InIterT& in_end,
                   OutIterT& out_beg) = 0;
};
} // namespace bgm

#endif // !BGM_DETECTION_FRAMEWORK_NMS_HPP_
