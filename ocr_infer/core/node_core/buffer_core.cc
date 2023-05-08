#include "ocr_infer/core/node_core/buffer_core.h"

#include "glog/logging.h"

BufferCore::BufferCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Buffer node init...";
  rec_batch_size_ = std::stoi(Inquire(config, "reco_batch_size"));
  LOG(INFO) << "rec_batch_size_ = " << rec_batch_size_;
  LOG(INFO) << "Buffer node init over!";
}

void BufferCore::Process(const std::shared_ptr<RecInput> &in,
                         std::vector<std::shared_ptr<RecInput>> *out_v) {
  VLOG(1) << "*** Buffer node, in size = " << in->clips.size();
  for (int i = 0; i < in->clips.size(); i++) {
    buffer_clips_.push(in->clips[i]);  // TODO: 存在拷贝，考虑使用引用 or 指针
    buffer_names_.push(in->names[i]);
    buffer_boxnum_.push(in->boxnum[i]);
    buffer_boxes_.push(in->boxes[i]);
  }
  while (buffer_clips_.size() > rec_batch_size_) {
    auto out = std::make_shared<RecInput>();
    for (int j = 0; j < rec_batch_size_; j++) {
      out->clips.emplace_back(Preprocess(buffer_clips_.front()));
      out->names.emplace_back(buffer_names_.front());
      out->boxnum.emplace_back(buffer_boxnum_.front());
      out->boxes.emplace_back(buffer_boxes_.front());
      buffer_clips_.pop();
      buffer_names_.pop();
      buffer_boxnum_.pop();
      buffer_boxes_.pop();
    }
    out_v->emplace_back(out);
  }
  VLOG(1) << "*** Buffer node, out_v size = " << out_v->size();
}

// TODO: 考虑将预处理放入识别节点，让 buffer 节点的功能更简单一点
cv::Mat BufferCore::Preprocess(const cv::Mat &input_image) {
  cv::Mat im_to_use;
  if (input_image.channels() == 3) {
    cvtColor(input_image, im_to_use, cv::COLOR_BGR2GRAY);
  } else {
    im_to_use = input_image.clone();
  }
  if (im_to_use.depth() != CV_32FC1) {
    im_to_use.convertTo(im_to_use, CV_32FC1);
  }
  return im_to_use;
}
