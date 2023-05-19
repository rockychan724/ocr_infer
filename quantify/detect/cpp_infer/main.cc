#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "db.h"
#include "db_pp.h"
#include "drawer.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

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

void ReadImages(std::vector<std::string> &file_names,
                std::vector<cv::Mat> &origin_images,
                std::vector<cv::Mat> &images,
                std::vector<cv::Point2f> &scales) {
  std::vector<cv::String> files;
  cv::glob("../../../testdata/e2e/image/*.jpg", files, false);
  size_t num = files.size();
  for (size_t i = 0; i < num; i++) {
    cv::Mat origin_img = cv::imread(files[i], cv::IMREAD_COLOR);
    int col = origin_img.cols, row = origin_img.rows;
    cv::Mat img;
    cv::resize(origin_img, img, cv::Size(512, 512));
    cv::Point2f scale_rate(512.0 / col, 512.0 / row);
    // std::cout << "img dim: " << img.dims << std::endl;
    // std::cout << "img channels: " << img.channels() << std::endl;
    // std::cout << "img size: " << img.size << std::endl;
    if (img.depth() != CV_32FC3) {
      // std::cout << "convert color " << img.depth() << " to " << CV_8UC4 <<
      // std::endl;
      img.convertTo(img, CV_32FC3);
      // cv::cvtColor(img, img, CV_BGR2BGRA);
    }
    int index1 = files[i].find_last_of("/");
    int index2 = files[i].find_last_of(".");
    file_names.push_back(files[i].substr(index1 + 1, index2 - index1 - 1));
    origin_images.push_back(origin_img);
    images.push_back(img);
    scales.push_back(scale_rate);
  }
}

void mat_to_txt(const cv::Mat &mat, const std::string &img_name, int j) {
  std::stringstream ss;
  ss << "./inference_output/probs/" << img_name << "_" << j << ".txt";
  std::ofstream f(ss.str());
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      f << mat.at<float>(i, j) << " ";
    }
    f << std::endl;
  }
  f.close();
}

int main(int argc, char **argv) {
  // google::InitGoogleLogging(argv[0]);
  // std::string log_dir = "log";
  // FLAGS_log_dir = log_dir;
  // system("mkdir -p log");

  system("rm -r ./inference_output");
  system(
      "mkdir -p ./inference_output/vis ./inference_output/preds "
      "./inference_output/probs");

  std::string model_path = "../../weights/trt_engine/3090/db_resnet_50.fp16";
  int batch_size = 50;

  Db detector(model_path, batch_size);
  DbPostprocessing db_pp;
  Drawer drawer;

  std::vector<std::string> file_names;
  std::vector<cv::Mat> origin_images;
  std::vector<cv::Mat> images;
  std::vector<cv::Point2f> scales;
  ReadImages(file_names, origin_images, images, scales);

  int batch_num = ceil(double(images.size()) / batch_size);
  size_t begin_index = 0;
  double tick_start, tick_end, total_infer_time = 0.0;
  for (int i = 0; i < batch_num; i++) {
    // begin = begin + batch_size >= images.size() ? 0 : begin + batch_size;
    int end_index = begin_index + batch_size >= images.size()
                        ? images.size()
                        : begin_index + batch_size;
    std::vector<std::string> sub_names(file_names.begin() + begin_index,
                                       file_names.begin() + end_index);
    std::vector<cv::Mat> sub_images(images.begin() + begin_index,
                                    images.begin() + end_index);
    std::vector<cv::Mat> sub_origin_images(origin_images.begin() + begin_index,
                                           origin_images.begin() + end_index);
    std::vector<cv::Point2f> sub_scales(scales.begin() + begin_index,
                                        scales.begin() + end_index);
    begin_index = end_index;

    std::vector<cv::Mat> pred_map;
    tick_start = Timer::GetMillisecond();
    detector.Forward(sub_images, &pred_map);
    total_infer_time += Timer::GetMillisecond() - tick_start;

    for (size_t j = 0; j < pred_map.size(); j++) {
      std::vector<cv::RotatedRect> boxes;
      db_pp.Parse(pred_map[j], sub_scales[j], &boxes);
      drawer.Draw(sub_origin_images[j], boxes, sub_names[j], true);
      // mat_to_txt(pred_map[j], sub_names[j], j);
    }
  }

  double average_time = total_infer_time / images.size();
  double fps = 1.0e3 / average_time;
  std::cout << "Test frames = " << images.size() << "\n"
            << "Total time = " << average_time / 1.0e3 << " s\n"
            << "Average time per image = " << average_time << " ms/image\n"
            << "FPS = " << fps << "\n";

  return 0;
}
