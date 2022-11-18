#ifndef OCR_INFER_TEST_TEST_SPEED_H_
#define OCR_INFER_TEST_TEST_SPEED_H_

#include "ocr_infer/core/pipeline/pipeline.h"

class TestSpeed {
 public:
  TestSpeed(const std::string &, int) {
    auto pipeline_io = PipelineFactory::BuildE2e({});
    sender_ = pipeline_io.first, receiver_ = pipeline_io.second;

    consumer_ = std::make_shared<std::thread>([this]() { Consume(); });
  }

  void Run() {
    while (1) {
      auto data = std::make_shared<DetInput>();
      sender_->push(data);
    }
  }

 private:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::shared_ptr<std::thread> consumer_;

  void Consume() {
    while (1) {
      receiver_->pop();
    }
  }
};

#endif  // OCR_INFER_TEST_TEST_SPEED_H_
