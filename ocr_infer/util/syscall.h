#ifndef OCR_INFER_UTIL_SYSCALL_H_
#define OCR_INFER_UTIL_SYSCALL_H_

// #if WIN32
// #include "ios.h"
// #else
#include <sys/stat.h>
#include <sys/unistd.h>
// #endif

static int Access(const char* name, int type) {
  return access(name, type);
}

static int Mkdir(const char* name, int mode) {
  return mkdir(name, mode);
}

#endif  // OCR_INFER_UTIL_SYSCALL_H_
