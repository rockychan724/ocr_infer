#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <unordered_map>

#include "glog/logging.h"
#include "ocr_infer/test/test_speed.h"

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

  LOG(INFO) << "run";

  TestSpeed test("", 0);
  test.Run();

  return 0;
}
