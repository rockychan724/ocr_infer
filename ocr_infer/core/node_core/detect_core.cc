#include "ocr_infer/core/node_core/detect_core.h"

#include "glog/logging.h"

DetectCore::DetectCore(const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Detect node init over!";
}

std::shared_ptr<DetOutput> DetectCore::Process(const std::shared_ptr<DetInput> &in) {
  return {};
}
