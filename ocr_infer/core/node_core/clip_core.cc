#include "ocr_infer/core/node_core/clip_core.h"

#include "glog/logging.h"

ClipCore::ClipCore(const std::unordered_map<std::string, std::string> &config)
    : rec_input_size_(480, 48) {
  LOG(INFO) << "Clip node init...";
  LOG(INFO) << "Clip node init over!";
}

std::shared_ptr<RecInput> ClipCore::Process(const std::shared_ptr<DetBox> &in) {
  VLOG(1) << "*** Clip node, in size = " << in->images.size();
  auto out = std::make_shared<RecInput>();
  for (int i = 0; i < in->images.size(); i++) {
    for (const auto &bb : in->boxes[i]) {
      try {
        // TODO: Add old method
        cv::Mat clip = this->GetRotateCropImage(in->images[i], bb, rec_input_size_);
        out->clips.emplace_back(clip);
        out->names.emplace_back(in->names[i]);
        out->box_num.emplace_back(in->boxes[i].size());
        out->boxes.emplace_back(bb);
      } catch (std::exception &e) {
        LOG(WARNING) << "An exception occurred when cropping " << in->names[i];
        LOG(WARNING) << e.what();
      }
    }
    VLOG(1) << "box num = " << in->boxes[i].size();
  }
  VLOG(1) << "*** Clip node, out size = " << out->clips.size();
  return out;
}

cv::Mat ClipCore::GetRotateCropImage(const cv::Mat &src_image, const cv::RotatedRect &box,
                                     const cv::Size &s) {
  // points[0]指的就是y值最大的坐标点，若最大的y值有两个，则左侧的（x较小的）为points[0]点，（坐标系原点在左上角，x轴水平向右，y轴竖直向下）。
  // width边是x轴逆时针旋转最先平行的边。
  // box的角度取值范围为[-90, 0]
  cv::Point2f points[4];
  box.points(points);

  cv::Rect rect = box.boundingRect();  // rect.x 和 rec.y 是外接矩形框左上点的x,y坐标

  // 获取box的四个点在外界矩形中的相对坐标
  for (auto &point : points) {
    point.x -= static_cast<float>(rect.x);
    point.y -= static_cast<float>(rect.y);
  }

  // 计算上下左右四个边界发生溢出时需要padding的值
  cv::Point top_left = {rect.x, rect.y};
  cv::Point bottom_right = {rect.x + rect.width, rect.y + rect.height};
  int edge_overflow[4];  // 上下左右四个边界
  edge_overflow[0] = std::max({0, 0 - top_left.y, 0 - bottom_right.y});
  edge_overflow[1] =
      std::max({0, top_left.y - (src_image.rows - 1), bottom_right.y - (src_image.rows - 1)});
  edge_overflow[2] = std::max({0, 0 - top_left.x, 0 - bottom_right.x});
  edge_overflow[3] =
      std::max({0, top_left.x - (src_image.cols - 1), bottom_right.x - (src_image.cols - 1)});

  cv::Mat padding_image;
  cv::Scalar padding_value;
  padding_value = src_image.channels() == 3 ? cv::Scalar(0, 0, 0) : cv::Scalar(0);
  cv::copyMakeBorder(src_image, padding_image, edge_overflow[0], edge_overflow[1], edge_overflow[2],
                     edge_overflow[3], cv::BORDER_CONSTANT, padding_value);  // padding

  // 更新rect的坐标，将rect的坐标对应到padding_image上
  if (rect.x < 0)
    rect.x = 0;
  else if (rect.x >= src_image.cols)
    rect.x = padding_image.cols;
  if (rect.y < 0)
    rect.y = 0;
  else if (rect.y >= src_image.rows)
    rect.y = padding_image.rows;

  cv::Mat img_crop;
  padding_image(rect).copyTo(img_crop);

  // 根据box的角度来判断rect的四个点的顺序以及宽和高的大小
  float box_width;
  float box_height;
  cv::Point2f pts_std[4];
  if (box.angle < -45) {  // [-90, -45) 倾斜过大导致h和w的大小互换了，points[0]为box右下角的点
    box_width = box.size.height;
    box_height = box.size.width;
    pts_std[0] = cv::Point2f(box_width, box_height);
    pts_std[1] = cv::Point2f(0, box_height);
    pts_std[2] = cv::Point2f(0, 0);
    pts_std[3] = cv::Point2f(box_width, 0);
  } else {  // [-45, 0]
    box_width = box.size.width;
    box_height = box.size.height;
    pts_std[0] = cv::Point2f(0, box_height);
    pts_std[1] = cv::Point2f(0, 0);
    pts_std[2] = cv::Point2f(box_width, 0);
    pts_std[3] = cv::Point2f(box_width, box_height);
  }

  cv::Mat dst_img;
  cv::Mat M = cv::getPerspectiveTransform(points, pts_std);
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(static_cast<int>(box_width), static_cast<int>(box_height)),
                      cv::BORDER_REPLICATE);

  // 逆时针旋转竖排文本90°
  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, dst_img, 0);
  }

  // resize and pad lastly
  int src_height = dst_img.rows, src_width = dst_img.cols;
  int new_height = s.height;
  int new_width = int(float(new_height) / float(src_height) * float(src_width));
  new_width = std::min(new_width, s.width);
  cv::resize(dst_img, dst_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
  if (new_width < s.width)
    cv::copyMakeBorder(dst_img, dst_img, 0, 0, 0, s.width - new_width, cv::BORDER_CONSTANT,
                       padding_value);
  return dst_img;
}
