#ifndef OCR_INFER_API_DATA_TYPE_H_
#define OCR_INFER_API_DATA_TYPE_H_

#include <functional>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

typedef std::function<void(const std::string &, const cv::Mat &, void *)>
    CallbackFunc;

typedef int KeywordId;

struct Input {
  std::vector<std::string> names;
  std::vector<cv::Mat> images;
};

struct Output {
  std::vector<std::string> names;
  std::vector<size_t> boxnum;
  std::vector<std::vector<std::string>> multitext;
  std::vector<std::vector<cv::RotatedRect>> multiboxes;
  std::vector<KeywordId> hitid;
};

#endif  // OCR_INFER_API_DATA_TYPE_H_
