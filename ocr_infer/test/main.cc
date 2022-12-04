#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include "ocr_infer/test/test_speed.h"

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program << "CONFIG_FILE_PATH TEST_DATA_DIR\n";
}

// TODO:
// 1. 程序异常检测
// 2. 启动参数
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string log_dir = "log";
  FLAGS_log_dir = log_dir;

  if (access(log_dir.c_str(), F_OK) == -1) {
    if (mkdir(log_dir.c_str(), 0775) == -1) {
      std::stringstream ss;
      ss << "mkdir: can't mkdir named \"" << log_dir << "\"";
      perror(ss.str().c_str());
      return -1;
    }
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
