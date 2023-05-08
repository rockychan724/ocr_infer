#ifndef OCR_INFER_API_OCR_API_H_
#define OCR_INFER_API_OCR_API_H_

#include <string>
#include <memory>

// #include "ocr_infer/api/data_type.h"

class OcrInfer {
 public:
  explicit OcrInfer(const std::string &config_file, void *callback_func);

  int Run(const std::string &image_dir);

 private:
  std::shared_ptr<void> ocr_handle_;

  // // TODO:
  // int Run(const Input &in);
};

#endif  // OCR_INFER_API_OCR_API_H_
