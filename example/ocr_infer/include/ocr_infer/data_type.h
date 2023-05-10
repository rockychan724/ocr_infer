#ifndef OCR_INFER_API_DATA_TYPE_H_
#define OCR_INFER_API_DATA_TYPE_H_

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

struct Input {
    std::vector<std::string> names;
    std::vector<cv::Mat> images;

    Input() = default;

    Input(std::vector<std::string> n, std::vector<cv::Mat> i) : names(std::move(n)), images(std::move(i)) {}
};

struct Output {
    std::vector<std::string> names;
    std::vector<std::string> text;
    std::vector<int> boxnum;
    std::vector<cv::RotatedRect> boxes;
};

#endif  // OCR_INFER_API_DATA_TYPE_H_
