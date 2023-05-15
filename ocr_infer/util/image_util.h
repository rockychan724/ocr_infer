#ifndef OCR_INFER_UTIL_IMAGE_UTIL_H_
#define OCR_INFER_UTIL_IMAGE_UTIL_H_

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "opencv2/opencv.hpp"

static void ReadImages(const std::string &images_path,
                       std::vector<std::string> &names,
                       std::unordered_map<std::string, cv::Mat> &images) {
  std::vector<cv::String> files;
  cv::glob(images_path + "/*.jpg", files, false);
  std::shuffle(files.begin(), files.end(), std::default_random_engine(9));
  size_t count = files.size();
  auto get_file_name = [](const std::string &file_path) -> std::string {
    int index1 = file_path.find_last_of("/");
    int index2 = file_path.find_last_of(".");
    return file_path.substr(index1 + 1, index2 - index1 - 1);
  };
  for (size_t i = 0; i < count; i++) {
    try {
      cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
      std::string file_name = get_file_name(files[i]);
      names.emplace_back(file_name);
      images.insert({file_name, img});
    } catch (cv::Exception &e) {
      std::cout << "****** Read " << files[i] << " error!\n"
                << e.what() << std::endl;
    }
  }
}

static void DrawDetectBox(cv::Mat &image, const cv::RotatedRect &box, const cv::Point2f *vertices2f) {
  // cv::Point2f vertices2f[4];
  // box.points(vertices2f);
  cv::Point root_points[1][4];
  for (int i = 0; i < 4; ++i) {
    root_points[0][i] = vertices2f[i];
    cv::putText(image, std::to_string(i), vertices2f[i], cv::FONT_HERSHEY_PLAIN,
            1.0, cv::Scalar(0, 255, 0), 2);  // debug
  }
  cv::putText(image, std::to_string(box.angle),
          cv::Point(box.center.x + box.size.width / 2.0, box.center.y),
          cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 2);  // debug

  // draw box
  const cv::Point *ppt[1] = {root_points[0]};
  int npt[] = {4};
  cv::polylines(image, ppt, npt, 1, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
}

#endif  // OCR_INFER_UTIL_IMAGE_UTIL_H_
