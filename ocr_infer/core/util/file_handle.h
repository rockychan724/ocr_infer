#ifndef OCR_INFER_CORE_UTIL_FILE_HANDLE_H_
#define OCR_INFER_CORE_UTIL_FILE_HANDLE_H_

#include <string>
#include <vector>

std::vector<std::string> GetFilesV1(std::string dir,
                                    const std::string &extension,
                                    bool is_recursive, bool with_path);

/**
 * @brief get files under the pattern by glob function
 *
 * @param pattern "/some/path/keyword_*.txt"
 * @return std::vector<std::string>
 */
std::vector<std::string> GetFilesV2(std::string dir,
                                    const std::string &extension);

#endif  // OCR_INFER_CORE_UTIL_FILE_HANDLE_H_
