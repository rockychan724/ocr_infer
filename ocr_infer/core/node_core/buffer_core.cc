#include "ocr_infer/core/node_core/buffer_core.h"

BufferCore::BufferCore(const std::unordered_map<std::string, std::string> &config) {}

void BufferCore::Process(const std::shared_ptr<RecInput> &in,
                         std::vector<std::shared_ptr<RecInput>> *out) {
  std::cout << "=== Buffer core process ===\n";
  out->push_back(in);
}
