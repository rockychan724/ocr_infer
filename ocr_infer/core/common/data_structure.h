#ifndef OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_
#define OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "opencv2/opencv.hpp"

typedef int KeywordId;

// TODO:
// 1. 使用指针存储数据
// 2. 添加 id 属性（现在暂时把 name 当成 id）
struct DetInput {
  std::vector<std::string> names;
  std::vector<cv::Mat> images;
};

struct DetOutput {
  std::vector<std::string> names;
  std::vector<cv::Mat> images;
  std::vector<cv::Point2f> scales;
  std::vector<cv::Mat> results;
};

struct DetBox {
  std::vector<std::string> names;
  std::vector<cv::Mat> images;
  std::vector<std::vector<cv::RotatedRect>> boxes;
};

struct RecInput {
  std::vector<std::string> names;
  std::vector<cv::Mat> clips;
  std::vector<size_t> boxnum;
  std::vector<cv::RotatedRect> boxes;
};

struct RecOutput {
  std::vector<std::string> names;
  std::vector<std::string> text;
  std::vector<size_t> boxnum;
  std::vector<cv::RotatedRect> boxes;
};

struct OcrOutput {
  std::vector<std::string> names;
  std::vector<size_t> boxnum;
  std::vector<std::vector<std::string>> multitext;
  std::vector<std::vector<cv::RotatedRect>> multiboxes;
};

struct MatchOutput : public OcrOutput {
  // TODO: 考虑命中多个敏感词 std::vector<std::vector<KeywordId>> hitids;
  std::vector<KeywordId> hitid;

  MatchOutput() = default;
  MatchOutput(const OcrOutput &ocr_output) : OcrOutput(ocr_output) {}
};

#endif  // OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_
