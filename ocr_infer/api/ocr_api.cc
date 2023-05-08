#include "ocr_infer/api/ocr_api.h"

#include "glog/logging.h"
#include "ocr_infer/engines/parallel_engine.h"

void check_license() {
  // TODO: 检查时间戳
}

OcrInfer::OcrInfer(const std::string& config_file, void* callback_func) {
  check_license();
  auto runtime_ptr = std::make_shared<ParallelEngine>();
  CHECK(runtime_ptr->Init(config_file, callback_func) == 0) << "Init failed!";
  ocr_handle_ = std::static_pointer_cast<void>(runtime_ptr);
}

int OcrInfer::Run(const std::string& image_dir) {
  auto runtime_ptr = std::static_pointer_cast<ParallelEngine>(ocr_handle_);
  return runtime_ptr->Run(image_dir);
}

// int OcrInfer::Run(const Input& in) {
//   auto runtime_ptr = std::static_pointer_cast<ParallelEngine>(ocr_handle_);
//   return runtime_ptr->Run(in);
// }
