#include "ocr_infer/core/util/file_handle.h"

#include "gtest/gtest.h"

TEST(TestFileHandle, test_file_handle_0) {
  std::string dir_path =
      "/home/chenlei/Documents/cnc/configuration/cnc_fuzzymatch_without_opencv/config/TexStar_data/"
      "sensitive_oneWordoneFile/";
  auto res1 = GetFilesV1(dir_path, ".txt", false, false);
  auto res2 = GetFilesV2(dir_path + "/*.txt");
  for (size_t i = 0; i < res1.size(); ++i) {
    // ASSERT_STREQ(res1[i], res2[i]);
    ASSERT_EQ(res1[i], res2[i]);
  }
}
