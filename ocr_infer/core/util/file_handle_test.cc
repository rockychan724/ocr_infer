#include "ocr_infer/core/util/file_handle.h"

#include <iostream>

#include "gtest/gtest.h"

TEST(TestFileHandle, test_get_files) {
  std::string dir_path =
      "/home/ocr_infer/data/keyword_files";
  auto res1 = GetFilesV1(dir_path, "txt", false, true);
  auto res2 = GetFilesV2(dir_path, "txt");
  ASSERT_EQ(res1.size(), res2.size());
  for (size_t i = 0; i < res1.size(); ++i) {
    // std::cout << res1[i] << ", " << res2[i] << std::endl;
    ASSERT_EQ(res1[i], res2[i]);
  }
}

TEST(TestFileHandle, test_read_unicode_file) {
  std::string dir_path =
      "/home/ocr_infer/data/keyword_files";
  auto file_path = GetFilesV1(dir_path, "txt", false, true);
  std::vector<std::vector<std::wstring>> res;
  for (size_t i = 0; i < file_path.size(); i++) {
    auto context = ReadUnicodeFile(file_path[i]);
    if (!context.empty()) {
      res.emplace_back(context);
    } else {
      std::cout << file_path[i] << " is empty!" << std::endl;
    }
  }
  ASSERT_EQ(file_path.size(), res.size());
}
