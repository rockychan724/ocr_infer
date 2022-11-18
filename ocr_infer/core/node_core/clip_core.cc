#include "ocr_infer/core/node_core/clip_core.h"

ClipCore::ClipCore(const std::unordered_map<std::string, std::string> &config) {
  
}

std::shared_ptr<RecInput> ClipCore::Process(const std::shared_ptr<DetBox> &in) {
  std::cout << "=== Clip core process ===\n";
  return {};
}
