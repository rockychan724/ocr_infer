#ifndef OCR_INFER_UTIL_IMAGE_UTIL_H_
#define OCR_INFER_UTIL_IMAGE_UTIL_H_

#include <random>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "opencv2/opencv.hpp"

static std::vector<std::string> GetFileName(
    const std::vector<std::string> &files) {
  std::vector<std::string> names;
  for (const auto &file : files) {
    int index1 = file.find_last_of("/");
    int index2 = file.find_last_of(".");
    names.emplace_back(file.substr(index1 + 1, index2 - index1 - 1));
  }
  return names;
}

static size_t ReadImages(const std::string &images_path,
                         std::vector<cv::Mat> &images,
                         std::vector<std::string> &names) {
  LOG(INFO) << "Begin reading images.";
  std::vector<cv::String> files;
  cv::glob(images_path + "/*.jpg", files, false);
  std::shuffle(files.begin(), files.end(), std::default_random_engine(9));
  size_t count = files.size();
  for (size_t i = 0; i < count; i++) {
    cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
    images.emplace_back(img);
  }
  names = GetFileName(files);
  printf("\nThere are %lu images\n\n", count);
  return count;
}

#endif  // OCR_INFER_UTIL_IMAGE_UTIL_H_
