#ifndef OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_

#include <memory>
#include <vector>

template <typename IType, typename OType>
class CoreBase {
 public:
  virtual std::shared_ptr<OType> Process(const std::shared_ptr<IType> &in) = 0;
};

template <typename IType, typename OType>
class BufferCoreBase {
 public:
  virtual void Process(const std::shared_ptr<IType> &in,
                       std::vector<std::shared_ptr<OType>> *out) = 0;
};

#endif  // OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_
