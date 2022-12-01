#ifndef OCR_INFER_CORE_MODEL_DETECT_DB_PP_H_
#define OCR_INFER_CORE_MODEL_DETECT_DB_PP_H_

#include <vector>

#include "opencv2/opencv.hpp"

/**
 * @brief Postprocessing of DB
 */
class DbPostprocessing {
 public:
  // TODO: Add construct function to set thresh and box_thresh.

  void Parse(const cv::Mat &pred_map, const cv::Point2f &scales,
             std::vector<cv::RotatedRect> *out);

 private:
  // TODO:
  // 比较当前后处理模块和PaddleOCR后处理模块对精度的影响；
  // 发现thresh和box_thresh参数对最终的结果影响较大
  float thresh = 0.1;      // 0.3;     // 0.1f;
  float box_thresh = 0.2;  // 0.5; // 0.2f;
  int max_candicates = 100;
  int min_side = 3;

  cv::Mat PartialVectorToMat(const cv::Mat &v, int y_start, int y_end,
                             int x_start, int x_end);
  float GetMeanScore(const cv::Mat &pred_map, const cv::RotatedRect &box);
  std::vector<cv::Point> Unclip(const cv::RotatedRect &box,
                                float Unclip_ratio = 1.5);
  inline void MapToOriginImage(const cv::Point2f &scales,
                               std::vector<cv::Point> &contour);
};

#endif  // OCR_INFER_CORE_MODEL_DETECT_DB_PP_H_
