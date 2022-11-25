#include "ocr_infer/core/util/file_handle.h"

#include <dirent.h>
#include <glob.h>

#include <algorithm>
#include <cstring>
#include <sstream>

std::vector<std::string> GetFilesV1(std::string dir, const std::string &extension,
                                    bool is_recursive, bool with_path) {
  if (dir.back() != '/') {
    dir += "/";
  }
  std::vector<std::string> filenames;
  DIR *pDIR;
  struct dirent *entry;

  if (is_recursive) {  // 当处理子目录时，必须存储完整路径
    with_path = true;
  }

  std::string filename;
  if (pDIR = opendir(dir.c_str())) {
    while (entry = readdir(pDIR)) {
      filename = entry->d_name;
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
        continue;
      }
      if (entry->d_type == 4) {  // 子文件夹的处理
        if (is_recursive) {
          std::vector<std::string> vFiles4subfolder =
              GetFilesV1(dir + entry->d_name, extension, is_recursive, with_path);
          filenames.insert(filenames.end(), vFiles4subfolder.begin(), vFiles4subfolder.end());
        } else {
          continue;
        }
      }

      bool isMatch = false;    // 判断当前文件是否是需要的文件
      if (extension == "*") {  // 扩展名为"*"时，只要是文件就保留
        isMatch = true;
      } else if (filename.substr(filename.find_last_of(".") + 1) == extension) {
        isMatch = true;
      }

      if (isMatch) {
        if (with_path)
          filenames.emplace_back(dir + filename);
        else
          filenames.emplace_back(filename);
      }
    }
    closedir(pDIR);
  }
  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

/**
 * @brief get files under the pattern by glob function
 *
 * @param pattern "/some/path/keyword_*.txt"
 * @return std::vector<std::string>
 */
std::vector<std::string> GetFilesV2(std::string dir, const std::string &extension) {
  if (dir.back() != '/') {
    dir += "/";
  }
  std::string pattern = dir + "*." + extension;

  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // do the glob operation
  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    std::stringstream ss;
    ss << "glob() failed with return_value " << return_value << std::endl;
    throw std::runtime_error(ss.str());
  }

  // collect all the filenames into a std::list<std::string>
  std::vector<std::string> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.emplace_back(std::string(glob_result.gl_pathv[i]));
  }

  // cleanup
  globfree(&glob_result);

  // done
  return filenames;
}
