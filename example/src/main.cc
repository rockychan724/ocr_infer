#include "ocr_api.h"

#include <iostream>

void Show(const std::string &res, void *other) {
  std::cout << res << std::endl;
}

int main() {
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";
  OcrInfer of;
  of.Init(config_file, Show, nullptr);
  of.Run(image_dir);
  return 0;
}
