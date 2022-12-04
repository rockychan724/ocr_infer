#ifndef OCR_INFER_CORE_UTIL_CODE_CONVERT_H_
#define OCR_INFER_CORE_UTIL_CODE_CONVERT_H_

#include <algorithm>
#include <codecvt>  // std::codecvt_utf8_utf16
#include <locale>   // std::wstring_convert
#include <string>
#include <vector>

static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

static inline std::wstring Utf8ToUtf16(const std::string &str) {
  return converter.from_bytes(str);
}

static inline std::vector<std::wstring> BatchUtf8ToUtf16(
    const std::vector<std::string> &str_v) {
  std::vector<std::wstring> wstr_v(str_v.size());
  std::transform(
      str_v.begin(), str_v.end(), wstr_v.begin(),
      [](const std::string &str) { return converter.from_bytes(str); });
  return wstr_v;
}

static inline std::string Utf16ToUtf8(const std::wstring &wstr) {
  return converter.to_bytes(wstr);
}

static inline std::vector<std::string> BatchUtf16ToUtf8(
    const std::vector<std::wstring> &wstr_v) {
  std::vector<std::string> str_v(wstr_v.size());
  std::transform(
      wstr_v.begin(), wstr_v.end(), str_v.begin(),
      [](const std::wstring &wstr) { return converter.to_bytes(wstr); });
  return str_v;
}

#endif  // OCR_INFER_CORE_UTIL_CODE_CONVERT_H_
