#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "crnn.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

namespace fs = std::filesystem;

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
                std::vector<cv::Mat> &images) {
  std::vector<cv::String> files;
  cv::glob("../../../testdata/rec/sub_mixed_test/*.jpg", files, false);
  size_t num = files.size();
  cv::Size img_size(480, 48);
  for (size_t i = 0; i < num; i++) {
    cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);

    if (img.depth() != CV_32FC1) {
      img.convertTo(img, CV_32FC1);
    }

    // 逆时针旋转竖排文本90°
    if (float(img.rows) >= float(img.cols) * 1.5) {
      cv::Mat srcCopy = cv::Mat(img.rows, img.cols, img.depth());
      cv::transpose(img, srcCopy);
      cv::flip(srcCopy, img, 0);
    }

    int h = img.rows, w = img.cols, hnew = 48;
    int wnew = int(1.0f * hnew / h * w);
    if (wnew < 480) {
      cv::Mat imgt(img_size, img.type());
      imgt.setTo(cv::Scalar(0));
      cv::resize(img, img, cv::Size(wnew, hnew), 0, 0, cv::INTER_CUBIC);
      img.copyTo(imgt(cv::Rect(cv::Point(0, 0), img.size())));
      images.push_back(imgt);
    } else {
      cv::resize(img, img, img_size, 0, 0, cv::INTER_CUBIC);
      images.push_back(img);
    }

    int index1 = files[i].find_last_of("/");
    file_names.push_back(
        files[i].substr(index1 + 1, files[i].length() - index1 - 1));
  }
}

int main(int argc, char **argv) {
  // google::InitGoogleLogging(argv[0]);
  // std::string log_dir = "log";
  // FLAGS_log_dir = log_dir;
  // system("mkdir -p log");

  fs::path output_dir = "./inference_output";
  if (fs::exists(output_dir)) {
    CHECK(fs::remove_all(output_dir)) << "Can't delete " << output_dir;
  }
  CHECK(fs::create_directories(output_dir)) << "Can't create " << output_dir;

  std::string fname = "../../weights/trt_engine/3090/crnn_100.fp16";
  std::string dict_path = "../../../data/rec_dict/dict_cjke.txt";
  int batch_size = 100;
  Crnn crnn(fname, dict_path, batch_size);

  std::vector<std::string> file_names;
  std::vector<cv::Mat> images;
  ReadImages(file_names, images);

  std::ofstream ofs("./inference_output/rec_result.txt");
  if (!ofs.is_open()) {
    std::cout << "Can't open ./inference_output/rec_output.txt" << std::endl;
  }
  int batch_num = ceil(float(images.size()) / batch_size);
  size_t begin_index = 0;
  double tick_start, tick_end, total_infer_time = 0.0;
  for (int i = 0; i < batch_num; i++) {
    int end_index = begin_index + batch_size >= images.size()
                        ? images.size()
                        : begin_index + batch_size;
    std::vector<std::string> sub_names(file_names.begin() + begin_index,
                                       file_names.begin() + end_index);
    std::vector<cv::Mat> sub_images(images.begin() + begin_index,
                                    images.begin() + end_index);
    begin_index = end_index;

    std::vector<std::string> result;
    tick_start = Timer::GetMillisecond();
    crnn.Forward(sub_images, &result);
    total_infer_time += Timer::GetMillisecond() - tick_start;

    for (size_t i = 0; i < result.size(); i++) {
      std::string info = sub_names[i] + " " + result[i] + "\n";
      std::cout << info;
      ofs << info;
    }
  }
  ofs.close();

  double average_time = total_infer_time / images.size();
  double fps = 1.0e3 / average_time;
  std::cout << "Test frames = " << images.size() << "\n"
            << "Total time = " << average_time / 1.0e3 << " s\n"
            << "Average time per image = " << average_time << " ms/image\n"
            << "FPS = " << fps << "\n";

  return 0;
}
