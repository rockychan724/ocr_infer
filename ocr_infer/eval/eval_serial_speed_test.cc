#include "ocr_infer/eval/eval_serial_speed.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestEvalSerialSpeed, test_eval_serial_speed) {
  EvalSerialSpeed eval_serial_speed;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";

  ASSERT_EQ(eval_serial_speed.Init(config_file, nullptr, nullptr), 0);
  ASSERT_EQ(eval_serial_speed.Run(image_dir), 0);
}
