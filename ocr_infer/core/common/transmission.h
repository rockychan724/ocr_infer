#ifndef OCR_INFER_CORE_COMMON_TRANSMISSION_H_
#define OCR_INFER_CORE_COMMON_TRANSMISSION_H_

#include <memory>

#include "tbb/concurrent_queue.h"

template <typename DType>
class QueueBase {
 protected:
  typedef tbb::concurrent_bounded_queue<std::shared_ptr<DType>> TbbQueue;

  std::shared_ptr<TbbQueue> q_;

  QueueBase(const std::shared_ptr<TbbQueue> &q) : q_(q) {}
};

/**
 * 在继承模板类时，需要注意一下几点：
 * 1. 子类构造函数调用父类构造函数时，如何调用父类的构造函数值得注意
 *    例如，下面子类构造函数调用父类构造函数形如 QueueBase<DType>(q)，而不是
 *    QueueBase(q)，QueueBase 是类模板，而 QueenBase<DTpye> 才是具体的类。
 * 2. 子类访问父类的成语变量或成员函数时，需要通过 this-> 或者
 *    QueenBase<DTpye>:: 来访问，或者在子类中添加 "using QueenBase<DTpye>::q_;"
 *    来显示说明子类中要访问父类哪些成员
 */
template <typename DType>
class QueueSender : public QueueBase<DType> {
 public:
  typedef tbb::concurrent_bounded_queue<std::shared_ptr<DType>> TbbQueue;

  QueueSender(const std::shared_ptr<TbbQueue> &q) : QueueBase<DType>(q) {}

  void push(const std::shared_ptr<DType> &data) {
    this->q_->push(data);  // 此处 this 不能省略
  }

  int size() { return this->q_->size(); }
};

template <typename DType>
class QueueReceiver : public QueueBase<DType> {
 public:
  typedef tbb::concurrent_bounded_queue<std::shared_ptr<DType>> TbbQueue;

  QueueReceiver(const std::shared_ptr<TbbQueue> &q) : QueueBase<DType>(q) {}

  std::shared_ptr<DType> pop() {
    std::shared_ptr<DType> res;
    this->q_->pop(res);  // TODO: 是否需要阻塞？还可考虑 try_pop()
    return res;
  }

  int size() { return this->q_->size(); }
};

template <typename DType>
class QueueFactory {
 public:
  typedef tbb::concurrent_bounded_queue<std::shared_ptr<DType>> TbbQueue;
  typedef std::pair<std::shared_ptr<QueueSender<DType>>,
                    std::shared_ptr<QueueReceiver<DType>>>
      QueuePair;

  static QueuePair BuildQueue(int capacity = 2) {
    std::shared_ptr<TbbQueue> q = std::make_shared<TbbQueue>();
    q->set_capacity(capacity);
    std::shared_ptr<QueueSender<DType>> sender =
        std::make_shared<QueueSender<DType>>(q);
    std::shared_ptr<QueueReceiver<DType>> receiver =
        std::make_shared<QueueReceiver<DType>>(q);
    return std::make_pair(sender, receiver);
  }
};

#endif  // OCR_INFER_CORE_COMMON_TRANSMISSION_H_
