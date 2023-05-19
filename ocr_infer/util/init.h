#ifndef OCR_INFER_UTIL_INIT_H_
#define OCR_INFER_UTIL_INIT_H_

#include <filesystem>
#include <unordered_map>

#include "glog/logging.h"

namespace fs = std::filesystem;

static int InitDirectory(const char *program, const fs::path &output_dir) {
#ifdef WRITE_LOG_TO_FILE
  google::InitGoogleLogging(program);
  std::string log_dir = "log";
  FLAGS_log_dir = log_dir;
  if (!fs::exists(log_dir)) {
    CHECK(fs::create_directories(log_dir)) << "Can't mkdir " << log_dir;
  }
#endif

  if (fs::exists(output_dir)) {
    CHECK(fs::remove_all(output_dir)) << "Can't delete " << output_dir;
  }
  CHECK(fs::create_directories(output_dir)) << "Can't create " << output_dir;

  return 0;
}

#endif  // OCR_INFER_UTIL_INIT_H_
