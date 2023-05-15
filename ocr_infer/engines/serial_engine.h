#ifndef OCR_INFER_ENGINES_SERIAL_ENGINE_H_
#define OCR_INFER_ENGINES_SERIAL_ENGINE_H_

#include <functional>
#include <memory>
#include <thread>

#include "ocr_infer/api/data_type.h"
#include "ocr_infer/core/pipeline/serial_pipeline.h"

typedef std::function<void(const std::string &, const cv::Mat &, void *)> CallbackFunc;

class SerialEngine {
 public:
  int Init(const std::string &config_file, CallbackFunc callback_func, void *other);

  int Run(const std::string &image_dir);

  std::string Run(const std::shared_ptr<Input> &in);

 private:
  std::unique_ptr<SerialE2ePipeline> serial_e2e_pipeline_;

  int detect_batch_size_;

  std::unordered_map<std::string, cv::Mat> images_;

  CallbackFunc callback_func_;
  void *other_;

  std::string Print(const std::shared_ptr<MatchOutput> &match_result, bool execute_callback_func=false);
};

#endif  // OCR_INFER_ENGINES_SERIAL_ENGINE_H_
