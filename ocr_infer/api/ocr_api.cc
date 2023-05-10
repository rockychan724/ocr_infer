#include "ocr_infer/api/ocr_api.h"

#include "glog/logging.h"
#include "ocr_infer/engines/parallel_engine.h"
#include "ocr_api.h"

void check_license() {
  // TODO: 检查时间戳
}

int OcrInfer::Init(const std::string& config_file, CallbackFunc callback_func, void *other) {
  check_license();
  auto runtime_ptr = std::make_shared<ParallelEngine>();
  CHECK(runtime_ptr->Init(config_file, callback_func, other) == 0) << "Init failed!";
  ocr_handle_ = std::static_pointer_cast<void>(runtime_ptr);
  return 0;
}

int OcrInfer::Run(const std::string& image_dir) {
  auto runtime_ptr = std::static_pointer_cast<ParallelEngine>(ocr_handle_);
  return runtime_ptr->Run(image_dir);
}

// int OcrInfer::Run(const Input& in) {
//   auto runtime_ptr = std::static_pointer_cast<ParallelEngine>(ocr_handle_);
//   return runtime_ptr->Run(in);
// }
