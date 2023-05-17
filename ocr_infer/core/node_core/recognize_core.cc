#include "ocr_infer/core/node_core/recognize_core.h"

#include <thread>

#include "glog/logging.h"

RecognizeCore::RecognizeCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Recognize node init...";
  recognizer_num_ = std::stoi(Inquire(config, "recognizer_num"));
  int rec_batch_size = std::stoi(Inquire(config, "rec_batch_size"));
  LOG(INFO) << "recognizer_num_ = " << recognizer_num_
            << ", rec_batch_size = " << rec_batch_size;

  std::string model_path = Inquire(config, "rec_model");
  std::string dict_path = Inquire(config, "dict");

  recognizer_.resize(recognizer_num_);
  for (int i = 0; i < recognizer_num_; i++) {
    recognizer_[i] =
        std::make_unique<Crnn>(model_path, dict_path, rec_batch_size);
  }

  LOG(INFO) << "Recognize node init over!";
}

std::shared_ptr<RecOutput> RecognizeCore::Process(
    const std::shared_ptr<RecInput> &in) {
  VLOG(1) << "*** Recognize node, in size = " << in->clips.size();
  if (in->clips.size() == 0) {
    return {};
  }

  // preprocess before recognize inference
  for (int i = 0; i < in->clips.size(); i++) {
    in->clips[i] = Preprocess(in->clips[i]);
  }

  auto out = std::make_shared<RecOutput>();
  out->names.assign(in->names.begin(), in->names.end());
  out->boxnum.assign(in->boxnum.begin(), in->boxnum.end());
  out->boxes.assign(in->boxes.begin(), in->boxes.end());

  std::vector<std::unique_ptr<std::thread>> threads;
  int one_batch = in->clips.size() / recognizer_num_;
  std::vector<std::vector<std::string>> output_result;
  output_result.resize(recognizer_num_);
  for (int i = 0; i < recognizer_num_; i++) {
    std::vector<cv::Mat> sub_image(in->clips.begin() + i * one_batch,
                                   in->clips.begin() + (i + 1) * one_batch);
    threads.emplace_back(
        std::make_unique<std::thread>([this, &output_result, i, sub_image]() {
          this->recognizer_[i]->Forward(sub_image, &output_result[i]);
        }));
  }
  for (int i = 0; i < recognizer_num_; i++) {
    threads[i]->join();
    if (!output_result[i].empty()) {
      std::for_each(output_result[i].begin(), output_result[i].end(),
                    [](const std::string &item) { VLOG(1) << item; });
      out->text.insert(out->text.end(), output_result[i].begin(),
                       output_result[i].end());
    }
  }

  VLOG(1) << "*** Recognize node, out size = " << out->text.size();
  return out;
}

// TODO: 可以放到 clip 节点中
cv::Mat RecognizeCore::Preprocess(const cv::Mat &input_image) {
  cv::Mat im_to_use;
  if (input_image.channels() == 3) {
    cvtColor(input_image, im_to_use, cv::COLOR_BGR2GRAY);
  } else {
    im_to_use = input_image.clone();  // TODO: 是否有必要复制一份
  }
  if (im_to_use.depth() != CV_32FC1) {
    im_to_use.convertTo(im_to_use, CV_32FC1);
  }
  return im_to_use;
}
