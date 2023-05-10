#ifndef OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
#define OCR_INFER_ENGINES_PARALLEL_ENGINE_H_

#include <functional>
#include <memory>
#include <thread>

#include "ocr_infer/api/data_type.h"
#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/common/transmission.h"

typedef std::function<void(const std::string &, void *)> CallbackFunc;

class ParallelEngine {
 public:
  ParallelEngine();

  ~ParallelEngine();

  int Init(const std::string &config_file, CallbackFunc callback_func, void *other);

  int Run(const std::string &image_dir);

 private:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::shared_ptr<std::thread> consumer_;
  bool stop_consume_;

  int detect_batch_size_;

  // void (*callback_func_)(const std::string &, void *);
  CallbackFunc callback_func_;
  void *other_;

  void Consume();

  void Print(const std::shared_ptr<MatchOutput> &match_result);
};

#endif  // OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
