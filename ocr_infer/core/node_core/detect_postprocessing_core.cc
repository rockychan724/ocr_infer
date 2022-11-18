#include "ocr_infer/core/node_core/detect_postprocessing_core.h"

DetectPostProcessingCore::DetectPostProcessingCore(
    const std::unordered_map<std::string, std::string> &config) {
  std::cout << "*** Detect postprocessing core ***\n";
}

std::shared_ptr<DetBox> DetectPostProcessingCore::Process(const std::shared_ptr<DetOutput> &in) {
  std::cout << "=== Detect postprocessing core process ===\n";
  return {};
}
