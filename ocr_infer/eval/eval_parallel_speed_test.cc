#include "ocr_infer/eval/eval_parallel_speed.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

// When testing speed, please build project under Release mode.
TEST(TestEvalParallelSpeed, test_eval_parallel_speed) {
  EvalParallelSpeed eval_parallel_speed;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";

  ASSERT_EQ(eval_parallel_speed.Init(config_file, nullptr, nullptr), 0);
  LOG(INFO) << "Begin to test speed, please wait...";
  ASSERT_EQ(eval_parallel_speed.Run(image_dir, 5000, 10000, 12000), 0);
}
