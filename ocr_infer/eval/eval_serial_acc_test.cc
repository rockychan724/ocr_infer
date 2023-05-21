#include "ocr_infer/eval/eval_serial_acc.h"

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestEvalSerialAcc, test_eval_serial_acc) {
  EvalSerialAcc eval_acc_handle;
  std::string config_file =
      "/home/ocr_infer/data/config.ini";
  std::string image_dir =
      "/home/ocr_infer/testdata/e2e/image/";

  ASSERT_EQ(eval_acc_handle.Init(config_file, nullptr, nullptr), 0);
  ASSERT_EQ(eval_acc_handle.Run(image_dir), 0);
}
