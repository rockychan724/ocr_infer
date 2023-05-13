# clipper
add_library(clipper OBJECT third_party/clipper.cc)

# file_handle
add_library(file_handle OBJECT file_handle.cc)

# add_executable(test_file_handle file_handle_test.cc)
add_executable(test_file_handle $<TARGET_OBJECTS:file_handle> file_handle_test.cc)
target_link_libraries(test_file_handle
  file_handle
  -lgtest -lgtest_main
  -lglog
  -lpthread
)
# add_test(NAME test_file_handle COMMAND test_file_handle)

# code_convert
add_executable(test_code_convert code_convert_test.cc)
target_link_libraries(test_code_convert
  -lgtest -lgtest_main
  -lpthread
)
# add_test(NAME test_code_convert COMMAND test_code_convert)
