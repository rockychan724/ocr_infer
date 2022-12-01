# file_handle
add_library(file_handle file_handle.cc)
target_link_libraries(
  file_handle
  -lglog
)

add_executable(test_file_handle file_handle_test.cc)
target_link_libraries(
  test_file_handle
  file_handle
  -lgtest -lgtest_main
)
add_test(NAME test_file_handle COMMAND test_file_handle)

# clipper
add_library(clipper third_party/clipper.cc)
