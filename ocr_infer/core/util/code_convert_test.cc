#include "ocr_infer/core/util/code_convert.h"

#include "gtest/gtest.h"

TEST(TestCodeConvert, test_code_convert1) {
  std::string utf8_str = "Hello world! 你好世界！";
  std::wstring utf16_str = L"Hello world! 你好世界！";

  auto res1 = Utf8ToUtf16(utf8_str);
  auto res2 = Utf16ToUtf8(utf16_str);
  ASSERT_EQ(utf16_str, res1);
  ASSERT_EQ(utf8_str, res2);
}

TEST(TestCodeConvert, test_code_convert2) {
  std::vector<std::string> utf8_str_v = {"Hello world!", "你好世界！"};
  std::vector<std::wstring> utf16_str_v = {L"Hello world!", L"你好世界！"};

  auto res1 = BatchUtf8ToUtf16(utf8_str_v);
  auto res2 = BatchUtf16ToUtf8(utf16_str_v);
  ASSERT_EQ(utf16_str_v, res1);
  ASSERT_EQ(utf8_str_v, res2);
}
