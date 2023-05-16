#ifndef DRAW_H
#define DRAW_H

#include <fstream>
#include <memory>
#include <sstream>

#include "opencv2/opencv.hpp"

class Drawer {
 public:
  void Draw(cv::Mat images, std::vector<cv::RotatedRect> boxes,
            std::string file_core_name, bool save_det_result = true) {
    std::string dpath = "../inference_output/vis/" + file_core_name + ".jpg";
    cv::Mat dim = images.clone();
    std::stringstream ss;
    for (auto rr : boxes) {
      cv::Point2f vertices2f[4];
      cv::Scalar s{rand() % 255, rand() % 255, rand() % 255};
      this->DrawRotatedRectangle(dim, rr, s, vertices2f);
      for (const auto &ver : vertices2f) {
        ss << int(ver.x) << "," << int(ver.y) << ",";
      }
      ss.seekp(-1, std::ios::end);  // remove "," on the line end
      ss << "\n";
    }
    cv::imwrite(dpath, dim);
    if (save_det_result) {
      std::string txt = "../inference_output/preds/" + file_core_name + ".txt";
      std::ofstream ofs(txt.c_str());
      ofs << ss.str();
      ofs.close();
    }
  }

 private:
  void DrawRotatedRectangle(cv::Mat &image, cv::RotatedRect rotatedRectangle,
                            cv::Scalar color, cv::Point2f *vertices2f) {
    // We take the edges that OpenCV calculated for us
    rotatedRectangle.points(vertices2f);
    cv::Point root_points[1][4];
    for (int i = 0; i < 4; ++i) {
      root_points[0][i] = vertices2f[i];
    }
    const cv::Point *ppt[1] = {root_points[0]};
    int npt[] = {4};
    cv::polylines(image, ppt, npt, 1, 1, color, 3, 8, 0);
  }
};

#endif  // DRAW_H
