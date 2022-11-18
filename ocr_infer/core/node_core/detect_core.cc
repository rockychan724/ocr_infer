#include "ocr_infer/core/node_core/detect_core.h"

DetectCore::DetectCore(const std::unordered_map<std::string, std::string> &config) {
  std::cout << "*** Detect core ***\n";
}

std::shared_ptr<DetOutput> DetectCore::Process(const std::shared_ptr<DetInput> &in) {
  std::cout << "=== Detect core process ===\n";
  return {};
}
