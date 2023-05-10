#include "ocr_infer/engines/parallel_engine.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(TestParallelEngine, test_parallel_engine) {
  ParallelEngine pe;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config_cnc.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";

  auto callback_func = [](const std::string &out, void *other) {
    std::cout << out << std::endl;
  };
  ASSERT_EQ(pe.Init(config_file, callback_func, nullptr), 0);
  ASSERT_EQ(pe.Run(image_dir), 0);
}
