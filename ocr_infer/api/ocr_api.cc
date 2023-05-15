#include "ocr_infer/api/ocr_api.h"

#include "glog/logging.h"
#include "ocr_infer/engines/parallel_engine.h"
#include "ocr_infer/engines/serial_engine.h"

typedef ParallelEngine Engine;
// typedef SerialEngine Engine;

void check_license() {
  // TODO: 检查时间戳
}

int OcrInfer::Init(const std::string& config_file, CallbackFunc callback_func, void *other) {
  check_license();
  auto runtime_ptr = std::make_shared<Engine>();
  CHECK(runtime_ptr->Init(config_file, callback_func, other) == 0) << "Init failed!";
  ocr_handle_ = std::static_pointer_cast<void>(runtime_ptr);
  return 0;
}

int OcrInfer::Run(const std::string& image_dir) {
  auto runtime_ptr = std::static_pointer_cast<Engine>(ocr_handle_);
  return runtime_ptr->Run(image_dir);
}

// Only for serial engine
std::string OcrInfer::Run(const std::shared_ptr<Input>& in) {
  // auto runtime_ptr = std::static_pointer_cast<Engine>(ocr_handle_);
  // return runtime_ptr->Run(in);
}
