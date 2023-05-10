#ifndef OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/node_core/core_base.h"

class BufferCore : public BufferCoreBase<RecInput, RecInput> {
 public:
  BufferCore(const std::unordered_map<std::string, std::string> &config);

  void Process(const std::shared_ptr<RecInput> &in,
               std::vector<std::shared_ptr<RecInput>> *out_v) override;

 private:
  std::queue<std::string> buffer_names_;
  std::queue<cv::Mat> buffer_clips_;
  std::queue<int> buffer_boxnum_;
  std::queue<cv::RotatedRect> buffer_boxes_;
  int rec_batch_size_;

  // cv::Mat Preprocess(const cv::Mat &input_image);
};

#endif  // OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_
