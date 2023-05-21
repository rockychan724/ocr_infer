#include "ocr_infer/eval/eval_parallel_acc.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestEvalParallelAcc, test_eval_parallel_acc) {
  EvalParallelAcc eval_parallel_acc;
  std::string config_file =
      "/home/ocr_infer/data/config.ini";
  std::string image_dir =
      "/home/ocr_infer/testdata/e2e/image/";

  ASSERT_EQ(eval_parallel_acc.Init(config_file, nullptr, nullptr), 0);
  ASSERT_EQ(eval_parallel_acc.Run(image_dir, 500, 1500, 2000), 0);
}
