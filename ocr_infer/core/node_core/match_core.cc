#include "ocr_infer/core/node_core/match_core.h"

#include "glog/logging.h"

MatchCore::MatchCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Match node init...";

  // const std::string keyword_dir = Inquire(config, "keyword_dir");
  std::string root_path = Inquire(config, "root_path");
  if (root_path.back() != '/') {
    root_path += "/";
  }
  std::string keyword_dir = Inquire(config, "keyword_dir");
  keyword_dir = root_path + keyword_dir;
  matcher_engine_ = std::make_unique<MatcherEngine>(keyword_dir);

  LOG(INFO) << "Match node init over!";
}

std::shared_ptr<MatchOutput> MatchCore::Process(
    const std::shared_ptr<RecOutput> &in) {
  auto out = std::make_shared<MatchOutput>();

  for (int i = 0; i < in->names.size(); i++) {
    const std::string &key = in->names[i];
    if (out->name2box_num.find(key) == out->name2box_num.end()) {
      out->name2box_num.insert({key, in->box_num[i]});
    }
    out->name2text[key].emplace_back(in->text[i]);
    out->name2boxes[key].emplace_back(in->boxes[i]);
  }

  for (auto it = out->name2text.begin(); it != out->name2text.end(); it++) {
    out->name2hitid[it->first] = matcher_engine_->Match(it->second);
    // debug
    std::stringstream ss;
    std::for_each(it->second.begin(), it->second.end(), [&ss](const std::string &str){
      ss << str << "; ";
    });
    VLOG(1) << it->first << ", " << ss.str() << ", "
            << out->name2hitid[it->first];
  }

  return out;
}
