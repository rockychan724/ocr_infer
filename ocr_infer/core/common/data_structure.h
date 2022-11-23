#ifndef OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_
#define OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "opencv2/opencv.hpp"

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
  std::vector<int> box_num;
  std::vector<cv::RotatedRect> boxes;
};

struct RecOutput {
  std::vector<std::string> names;
  std::vector<std::string> text;
  std::vector<int> box_num;
  std::vector<cv::RotatedRect> boxes;
};

struct MatchOutput {
  std::unordered_map<std::string, int> name2clip_num;
  std::unordered_map<std::string, std::vector<std::string>> name2rec_result;
  std::unordered_map<std::string, long long> name2hitid;
  std::unordered_map<std::string, std::vector<cv::RotatedRect>> name2boxes;
};

#endif  // OCR_INFER_CORE_COMMON_DATA_STRUCTURE_H_
