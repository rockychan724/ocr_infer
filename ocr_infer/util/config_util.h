#ifndef OCR_INFER_UTIL_CONFIG_UTIL_H_
#define OCR_INFER_UTIL_CONFIG_UTIL_H_

#include <fstream>
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

static bool IsSpace(char c) {
  if (' ' == c || '\t' == c) return true;
  return false;
}

static bool IsCommentChar(char c) {
  switch (c) {
    case ';':
      return true;
    default:
      return false;
  }
}

static void Trim(std::string &str) {
  if (str.empty()) return;
  int i, start_pos, end_pos;
  for (i = 0; i < str.size(); i++) {
    if (!IsSpace(str[i])) break;
  }
  if (i == str.size()) {
    str = "";
    return;
  }
  start_pos = i;
  for (i = str.size() - 1; i >= 0; i--) {
    if (!IsSpace(str[i])) break;
  }
  end_pos = i;
  str = str.substr(start_pos, end_pos - start_pos + 1);
}

static bool AnalyseLine(const std::string &line, std::string &section,
                        std::string &key, std::string &value) {
  if (line.empty()) return false;
  int start_pos = 0, end_pos = line.size() - 1, pos, s_startpos, s_endpos;
  if ((pos = line.find(";")) != -1) {
    if (0 == pos) return false;
    end_pos = pos - 1;
  }
  if (((s_startpos = line.find("[")) != -1) &&
      ((s_endpos = line.find("]"))) != -1) {
    section = line.substr(s_startpos + 1, s_endpos - 1);
    return true;
  }
  std::string new_line = line.substr(start_pos, start_pos + 1 - end_pos);
  if ((pos = new_line.find('=')) == -1) return false;
  key = new_line.substr(0, pos);
  value = new_line.substr(pos + 1, end_pos + 1 - (pos + 1));
  Trim(key);
  if (key.empty()) return false;
  Trim(value);
  if ((pos = value.find("\r")) > 0) value.replace(pos, 1, "");
  if ((pos = value.find("\n")) > 0) value.replace(pos, 1, "");
  return true;
}

static bool ReadConfig(
    const std::string &config_file, const std::string &section,
    std::unordered_map<std::string, std::string> &config_map) {
  std::ifstream infile(config_file.c_str());
  if (!infile) return false;
  std::string line, key, value, _section;
  std::unordered_map<std::string, std::string> k_v;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      settings;
  while (getline(infile, line)) {
    if (AnalyseLine(line, _section, key, value)) {
      auto it = settings.find(_section);
      if (it != settings.end()) {
        k_v[key] = value;
        it->second = k_v;
      } else {
        k_v.clear();
        settings.insert(make_pair(_section, k_v));
      }
    }
    key.clear();
    value.clear();
  }
  infile.close();
  auto it = settings.find(section);
  if (it == settings.end())
    return false;
  else
    config_map = it->second;
  return true;
}

#endif  // OCR_INFER_UTIL_CONFIG_UTIL_H_
