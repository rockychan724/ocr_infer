#ifndef OCR_INFER_API_DATA_TYPE_H_
#define OCR_INFER_API_DATA_TYPE_H_

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

typedef int KeywordId;

struct Input {
  std::vector<std::string> names;
  std::vector<cv::Mat> images;

  Input() = default;

  Input(std::vector<std::string> n, std::vector<cv::Mat> i)
      : names(std::move(n)), images(std::move(i)) {}
};

struct Output {
  std::unordered_map<std::string, int> name2boxnum;
  std::unordered_map<std::string, std::vector<std::string>> name2text;
  std::unordered_map<std::string, std::vector<cv::RotatedRect>> name2boxes;
  // TODO: 考虑命中多个敏感词 KeywordId -> std::vector<KeywordId>
  std::unordered_map<std::string, KeywordId> name2hitid;
};

#endif  // OCR_INFER_API_DATA_TYPE_H_
