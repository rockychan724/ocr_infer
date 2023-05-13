add_library(engines OBJECT
  parallel_engine.cc
  serial_engine.cc
)

add_executable(test_parallel_engine
  parallel_engine_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_parallel_engine
  core
  -lgtest -lgtest_main
)

add_executable(test_serial_engine
  serial_engine_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_serial_engine
  core
  -lgtest -lgtest_main
)
