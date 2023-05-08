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
  std::vector<int> boxnum;
  std::vector<cv::RotatedRect> boxes;
};

struct RecOutput {
  std::vector<std::string> names;
  std::vector<std::string> text;
  std::vector<int> boxnum;
  std::vector<cv::RotatedRect> boxes;
};

struct MatchOutput {
  std::unordered_map<std::string, int> name2boxnum;
  std::unordered_map<std::string, std::vector<std::string>> name2text;
  std::unordered_map<std::string, std::vector<cv::RotatedRect>> name2boxes;
  // TODO: 考虑命中多个敏感词 KeywordId -> std::vector<KeywordId>
  std::unordered_map<std::string, KeywordId> name2hitid;
};

#endif  // OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_
