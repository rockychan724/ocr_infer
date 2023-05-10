#ifndef OCR_INFER_UTIL_INIT_H_
#define OCR_INFER_UTIL_INIT_H_

#include "glog/logging.h"
#include "ocr_infer/util/syscall.h"

static int InitLog(const char *program) {
  google::InitGoogleLogging(program);
  std::string log_dir = "log";
  FLAGS_log_dir = log_dir;

  if (Access(log_dir.c_str(), 0) == -1) {
    if (Mkdir(log_dir.c_str(), 0775) == -1) {
      std::stringstream ss;
      ss << "mkdir: can't mkdir named \"" << log_dir << "\"";
      perror(ss.str().c_str());
      return -1;
    }
  }

  return 0;
}

#endif  // OCR_INFER_UTIL_INIT_H_
