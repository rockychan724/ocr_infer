#include "ocr_infer/engines/parallel_engine.h"

#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

void Print(const std::string &out, void *other) {
  std::cout << out << std::endl;
}

TEST(TestParallelEngine, test_parallel_engine) {
  ParallelEngine pe;
  std::string config_file =
      "/home/chenlei/Documents/cnc/ocr_infer/data/config_cnc.ini";
  std::string image_dir = "/home/chenlei/Documents/cnc/testdata/image/";
  ASSERT_EQ(pe.Init(config_file, Print, nullptr), 0);
  ASSERT_EQ(pe.Run(image_dir), 0);
}
