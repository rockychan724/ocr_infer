#ifndef OCR_INFER_CORE_NODE_NODE_H_
#define OCR_INFER_CORE_NODE_NODE_H_

#include <memory>
#include <thread>
#include <unordered_map>

#include "ocr_infer/core/common/transmission.h"
#include "ocr_infer/core/node_core/core_base.h"

template <typename IType, typename OType>
class NodeBase {
 public:
  // NodeBase(const std::shared_ptr<QueueReceiver<IType>> &in_queue,
  //          const std::shared_ptr<QueueSender<IType>> &out_queue)
  //     : in_queue_(in_queue), out_queue_(out_queue) {}

  void SetUp(const std::unordered_map<std::string, std::string> &config,
             const std::shared_ptr<QueueReceiver<IType>> &in_queue,
             const std::shared_ptr<QueueSender<OType>> &out_queue, const std::string &node_name) {
    in_queue_ = in_queue;
    out_queue_ = out_queue;
    node_name_ = node_name;
    InitCore(config);
    worker_ = std::make_shared<std::thread>([this]() { this->Run(); });
  }

 protected:
  std::string node_name_;
  std::shared_ptr<QueueReceiver<IType>> in_queue_;
  std::shared_ptr<QueueSender<OType>> out_queue_;
  std::shared_ptr<std::thread> worker_;

  virtual void InitCore(const std::unordered_map<std::string, std::string> &config) = 0;

  virtual void Run() = 0;
};

template <typename IType, typename OType, typename CType>
class Node : public NodeBase<IType, OType> {
 protected:
  std::unique_ptr<CoreBase<IType, OType>> core_;

  void InitCore(const std::unordered_map<std::string, std::string> &config) override {
    core_ = std::make_unique<CType>(config);
  }

  void Run() override {
    while (true) {
      auto in = this->in_queue_->pop();
      auto out = core_->Process(in);
      this->out_queue_->push(out);
    }
  }
};

template <typename IType, typename OType, typename CType>
class Buffer : public NodeBase<IType, OType> {
 protected:
  std::unique_ptr<BufferCoreBase<IType, OType>> core_;

  void InitCore(const std::unordered_map<std::string, std::string> &config) override {
    core_ = std::make_unique<CType>(config);
  }

  void Run() override {
    while (true) {
      auto in = this->in_queue_->pop();
      std::vector<std::shared_ptr<OType>> out_buff;
      core_->Process(in, &out_buff);
      for (const auto &o : out_buff) {
        this->out_queue_->push(o);
      }
    }
  }
};

#endif  // OCR_INFER_CORE_NODE_NODE_H_
