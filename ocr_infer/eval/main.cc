#include <iostream>
#include <string>
#include <unordered_map>

#include "ocr_infer/eval/test_speed.h"
#include "ocr_infer/util/init.h"

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program << "CONFIG_FILE_PATH TEST_DATA_DIR\n";
}

// TODO:
// 1. 程序异常检测
// 2. 启动参数
int main(int argc, char** argv) {
  int init_res = InitLog(argv[0]);
  if (init_res != 0) {
    return init_res;
  }

  std::string config_path, test_data_dir;
  if (argc == 1) {
    config_path =
        "/home/chenlei/Documents/cnc/ocr_infer/data/config_cnc.ini";
    test_data_dir = "/home/chenlei/Documents/cnc/testdata/image/";
  } else if (argc == 3) {
    config_path = argv[1];
    test_data_dir = argv[2];
  } else {
    PrintUsage(argv[0]);
    return -1;
  }

  LOG(INFO) << "run";

  TestSpeed test(config_path, 0);
  test.Run(test_data_dir);

  return 0;
}
