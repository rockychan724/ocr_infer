#include "ocr_infer/core/node_core/detect_postprocessing_core.h"

#include "glog/logging.h"

DetectPostProcessingCore::DetectPostProcessingCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Detect postprocessing node init over!";
}

std::shared_ptr<DetBox> DetectPostProcessingCore::Process(const std::shared_ptr<DetOutput> &in) {
  return {};
}
