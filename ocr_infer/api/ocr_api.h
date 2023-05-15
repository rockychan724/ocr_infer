#ifndef OCR_INFER_API_OCR_API_H_
#define OCR_INFER_API_OCR_API_H_

#include <memory>
#include <string>

#include "ocr_infer/api/data_type.h"

class OcrInfer {
 public:
  int Init(const std::string &config_file, CallbackFunc callback_func,
           void *other);

  int Run(const std::string &image_dir);

  // // Only for serial engine
  // std::string Run(const std::shared_ptr<Input> &in);

 private:
  std::shared_ptr<void> ocr_handle_;
};

#endif  // OCR_INFER_API_OCR_API_H_
