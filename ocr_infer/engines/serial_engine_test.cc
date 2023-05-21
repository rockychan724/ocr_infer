#include "ocr_infer/engines/serial_engine.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestSerialEngine, test_serial_engine) {
  SerialEngine serial_engine;
  std::string config_file =
      "/home/ocr_infer/data/config.ini";
  std::string image_dir =
      "/home/ocr_infer/testdata/e2e/image/";

  auto callback_func = [](const std::string &out, const cv::Mat &det_res,
                          void *other) { std::cout << out << std::endl; };
  ASSERT_EQ(serial_engine.Init(config_file, callback_func, nullptr), 0);
  ASSERT_EQ(serial_engine.Run(image_dir), 0);
}
