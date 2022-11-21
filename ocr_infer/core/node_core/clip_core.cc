#include "ocr_infer/core/node_core/clip_core.h"

#include "glog/logging.h"

ClipCore::ClipCore(const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Clip node init over!";
}

std::shared_ptr<RecInput> ClipCore::Process(const std::shared_ptr<DetBox> &in) {
  return {};
}
