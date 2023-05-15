#include "ocr_infer/eval/eval_acc.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestEvalAcc, test_eval_acc) {
  EvalAcc ea;
  std::string base_dir = "/home/chenlei/Documents/cnc/";
  std::string config_file = base_dir + "ocr_infer/data/config_cnc.ini";
  std::string image_dir = base_dir + "testdata/image/";
  std::string output_dir = base_dir + "output";

  ASSERT_EQ(ea.Init(config_file, nullptr, nullptr, output_dir), 0);
  LOG(INFO) << "Begin to test speed, please wait...";
  ASSERT_EQ(ea.Run(image_dir, 500, 1500, 2000), 0);
}
