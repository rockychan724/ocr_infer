#include "ocr_infer/eval/eval_acc.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestEvalAcc, test_eval_acc) {
  EvalAcc eval_acc_handle;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config_cnc.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";

  ASSERT_EQ(eval_acc_handle.Init(config_file, nullptr, nullptr), 0);
  ASSERT_EQ(eval_acc_handle.Run(image_dir, 500, 1500, 2000), 0);
}
