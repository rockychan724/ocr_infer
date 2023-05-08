#ifndef OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
#define OCR_INFER_ENGINES_PARALLEL_ENGINE_H_

#include <memory>
#include <thread>

#include "ocr_infer/api/data_type.h"
#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/common/transmission.h"

// typedef void (*func)(const std::string &str);  // 使用 c++ 的 function

class ParallelEngine {
 public:
  int Init(const std::string &config_file, void *callback_func);

  int Run(const std::string &image_dir);

  int Run(const Input &in);

 private:
  std::shared_ptr<QueueSender<DetInput>> sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> receiver_;
  std::shared_ptr<std::thread> consumer_;

  int detect_batch_size_;

  // func *callback_func_;
  void (*callback_func_)(const std::string &str);

  void GatherResult();
};

#endif  // OCR_INFER_ENGINES_PARALLEL_ENGINE_H_
