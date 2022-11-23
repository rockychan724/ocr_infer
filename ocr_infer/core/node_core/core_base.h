#ifndef OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

// TODO: 模板可以去掉
template <typename IType, typename OType>
class CoreBase {
 protected:
  // TODO: 更加智能，设置默认值（推荐值），若没有查询到，返回默认值，并输出 warning
  static std::string Inquire(const std::unordered_map<std::string, std::string> &config,
                             const std::string &key) {
    auto it = config.find(key);
    CHECK(it != config.end()) << "Can't find \"" << key << "\" in config!";
    return it->second;
  }
};

template <typename IType, typename OType>
class NodeCoreBase : public CoreBase<IType, OType> {
 public:
  virtual std::shared_ptr<OType> Process(const std::shared_ptr<IType> &in) = 0;
};

template <typename IType, typename OType>
class BufferCoreBase : public CoreBase<IType, OType> {
 public:
  virtual void Process(const std::shared_ptr<IType> &in,
                       std::vector<std::shared_ptr<OType>> *out_v) = 0;
};

#endif  // OCR_INFER_CORE_NODE_CORE_NODE_CORE_H_
