#include "ocr_infer/eval/eval_speed.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

// When testing speed, please build project under Release mode.
TEST(TestEvalSpeed, test_eval_speed) {
  EvalSpeed eval_speed_handle;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config_cnc.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";

  ASSERT_EQ(eval_speed_handle.Init(config_file, nullptr, nullptr), 0);
  LOG(INFO) << "Begin to test speed, please wait...";
  ASSERT_EQ(eval_speed_handle.Run(image_dir, 5000, 10000, 12000), 0);
}
