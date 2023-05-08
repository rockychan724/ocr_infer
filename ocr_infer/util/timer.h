#ifndef OCR_INFER_UTIL_TIMER_H_
#define OCR_INFER_UTIL_TIMER_H_

#include <ctime>

class Timer {
 public:
  static double GetMillisecond() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1.0e3 + ts.tv_nsec / 1000000.0;
  }

  static double GetMicrosecond() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1.0e6 + ts.tv_nsec / 1000.0;
  }

  static double GetNanosecond() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1.0e9 + ts.tv_nsec;
  }
};

#endif  // OCR_INFER_UTIL_TIMER_H_
