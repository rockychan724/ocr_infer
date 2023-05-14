#include "ocr_infer/core/node_core/gather_core.h"

GatherCore::GatherCore(
    const std::unordered_map<std::string, std::string>& config) {
  LOG(INFO) << "Gather node init...";
  LOG(INFO) << "Gather node init over!";
}

std::shared_ptr<OcrOutput> GatherCore::Process(
    const std::shared_ptr<RecOutput>& in) {
  auto out = std::make_shared<OcrOutput>();

  for (int i = 0; i < in->names.size(); i++) {
    if (buffer_text_.size() < in->boxnum[i]) {
      buffer_text_.emplace_back(in->text[i]);  // TODO: 存在拷贝，考虑使用引用 or 指针
      buffer_boxes_.emplace_back(in->boxes[i]);
    }

    if (buffer_text_.size() == in->boxnum[i]) {
      out->names.emplace_back(in->names[i]);
      out->boxnum.emplace_back(in->boxnum[i]);
      out->multitext.emplace_back(buffer_text_);
      out->multiboxes.emplace_back(buffer_boxes_);

      buffer_text_.clear();
      buffer_boxes_.clear();
    }
  }

  return out;
}
