#ifndef OCR_INFER_EVAL_EVAL_SPEED_H_
#define OCR_INFER_EVAL_EVAL_SPEED_H_

#include "ocr_infer/engines/parallel_engine.h"

class EvalSpeed : public ParallelEngine {
 protected:
  // dry run
  virtual void Consume() override {
    while (!stop_consume_) {
      auto match_result = receiver_->pop();
    }
  }
};

#endif  // OCR_INFER_EVAL_EVAL_SPEED_H_
