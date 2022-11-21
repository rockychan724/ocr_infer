#include "ocr_infer/core/node_core/buffer_core.h"

#include "glog/logging.h"

BufferCore::BufferCore(const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "BUffer node init over!";
}

void BufferCore::Process(const std::shared_ptr<RecInput> &in,
                         std::vector<std::shared_ptr<RecInput>> *out) {
  out->push_back(in);
}
