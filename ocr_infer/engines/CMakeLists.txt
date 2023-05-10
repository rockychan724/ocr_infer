add_library(engines parallel_engine.cc serial_engine.cc)
target_link_libraries(
  engines
  core
)

add_executable(test_parallel_engine parallel_engine_test.cc)
target_link_libraries(
  test_parallel_engine
  engines
  -lgtest -lgtest_main
)

add_executable(test_serial_engine serial_engine_test.cc)
target_link_libraries(
  test_serial_engine
  engines
  -lgtest -lgtest_main
)
