#include "ocr_infer/engines/serial_engine.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestSerialEngine, test_serial_engine) {
  SerialEngine serial_engine;
  std::string base_dir = "/home/chenlei/Documents/cnc/";
  std::string config_file = base_dir + "ocr_infer/data/config_cnc.ini";
  std::string image_dir = base_dir + "testdata/image/";

  auto callback_func = [](const std::string &out, const cv::Mat &det_res,
                          void *other) { std::cout << out << std::endl; };
  ASSERT_EQ(serial_engine.Init(config_file, callback_func, nullptr), 0);
  ASSERT_EQ(serial_engine.Run(image_dir), 0);
}
