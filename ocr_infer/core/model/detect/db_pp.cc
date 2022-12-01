#include "ocr_infer/core/model/detect/db_pp.h"

#include <algorithm>
#include <iostream>

#include "ocr_infer/core/util/third_party/clipper.hpp"
#include "opencv2/imgproc.hpp"

void DbPostprocessing::Parse(const cv::Mat &pred_map, const cv::Point2f &scales,
                             std::vector<cv::RotatedRect> *out) {
  cv::Mat mat;
  cv::threshold(pred_map, mat, thresh, 255, 0);
  mat.convertTo(mat, CV_8UC1);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  int num_contours = std::min(int(contours.size()), this->max_candicates);
  for (int i = 0; i < num_contours; i++) {
    cv::RotatedRect box1 = cv::minAreaRect(contours[i]);
    if (std::min(box1.size.height, box1.size.width) < this->min_side) {
      continue;
    }
    float mean_score = this->GetMeanScore(pred_map, box1);
    if (mean_score < this->box_thresh) {
      continue;
    }
    std::vector<cv::Point> contour = this->Unclip(box1);
    this->MapToOriginImage(scales, contour);
    cv::RotatedRect box2 = cv::minAreaRect(contour);
    cv::Point2f p[4];
    box2.points(p);
    if (std::min(box2.size.height, box2.size.width) < this->min_side + 2) {
      continue;
    }
    out->emplace_back(box2);
  }
}

// TODO: 优化加速
cv::Mat DbPostprocessing::PartialVectorToMat(const cv::Mat &v, int y_start,
                                             int y_end, int x_start,
                                             int x_end) {
  cv::Mat mat(y_end - y_start, x_end - x_start, CV_32FC1);
  for (int i = 0; i < mat.rows; i++)
    for (int j = 0; j < mat.cols; j++) {
      mat.at<float>(i, j) = v.at<float>(y_start + i, x_start + j);
    }
  return mat;
}

float DbPostprocessing::GetMeanScore(const cv::Mat &pred_map,
                                     const cv::RotatedRect &box) {
  int height = pred_map.rows, width = pred_map.cols;
  cv::Point2f points[4];
  box.points(points);
  float f_xmin =
      std::min_element(points, points + 3, [](cv::Point2f p1, cv::Point2f p2) {
        return p1.x < p2.x;
      })->x;
  float f_xmax =
      std::max_element(points, points + 3, [](cv::Point2f p1, cv::Point2f p2) {
        return p1.x < p2.x;
      })->x;
  float f_ymin =
      std::min_element(points, points + 3, [](cv::Point2f p1, cv::Point2f p2) {
        return p1.y < p2.y;
      })->y;
  float f_ymax =
      std::max_element(points, points + 3, [](cv::Point2f p1, cv::Point2f p2) {
        return p1.y < p2.y;
      })->y;
  int xmin = std::max(0, static_cast<int>(floor(f_xmin)));
  int xmax = std::min(width - 1, static_cast<int>(ceil(f_xmax)));
  int ymin = std::max(0, static_cast<int>(floor(f_ymin)));
  int ymax = std::min(height - 1, static_cast<int>(ceil(f_ymax)));
  cv::Mat mask(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1, cv::Scalar(0));
  cv::Point root_points[1][4];
  std::transform(points, points + 4, root_points[0], [=](cv::Point2f p) {
    cv::Point res;
    res.x = p.x - xmin;
    res.y = p.y - ymin;
    return res;
  });
  const cv::Point *pts[1] = {root_points[0]};
  int npts[] = {4};
  cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1));
  cv::Mat pred_mat =
      this->PartialVectorToMat(pred_map, ymin, ymax + 1, xmin, xmax + 1);
  cv::Scalar s = cv::mean(pred_mat, mask);
  return static_cast<float>(s[0]);
}

std::vector<cv::Point> DbPostprocessing::Unclip(const cv::RotatedRect &box,
                                                float Unclip_ratio) {
  double distance =
      box.size.area() * Unclip_ratio / ((box.size.height + box.size.width) * 2);
  cv::Point2f points[4];
  box.points(points);
  ClipperLib::Path path(4);
  std::transform(points, points + 4, path.begin(), [](cv::Point2f p) {
    return ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(p.x),
                                static_cast<ClipperLib::cInt>(p.y));
  });
  ClipperLib::Paths solution;
  ClipperLib::ClipperOffset offset;
  offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  offset.Execute(solution, distance);
  std::vector<cv::Point> res(solution[0].size());
  std::transform(solution[0].begin(), solution[0].end(), res.begin(),
                 [](ClipperLib::IntPoint p) {
                   return cv::Point(static_cast<int>(p.X),
                                    static_cast<int>(p.Y));
                 });
  return res;
}

inline void DbPostprocessing::MapToOriginImage(
    const cv::Point2f &scales, std::vector<cv::Point> &contour) {
  std::for_each(contour.begin(), contour.end(), [=](cv::Point &p) {
    p.x = p.x / scales.x;
    p.y = p.y / scales.y;
  });
}
