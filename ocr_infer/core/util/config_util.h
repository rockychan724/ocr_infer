#ifndef OCR_INFER_CORE_UTIL_CONFIG_UTIL_H_
#define OCR_INFER_CORE_UTIL_CONFIG_UTIL_H_

#include <string>
#include <unordered_map>

#include "glog/logging.h"

static std::string Query(
    const std::unordered_map<std::string, std::string> &config,
    const std::string &key) {
  auto it = config.find(key);
  CHECK(it != config.end()) << "Can't find \"" << key << "\" in config!";
  return it->second;
}

#endif  // OCR_INFER_CORE_UTIL_CONFIG_UTIL_H_
