add_executable(test_eval_parallel_speed
  eval_parallel_speed_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_parallel_speed
  core
  -lgtest -lgtest_main
)

add_executable(test_eval_parallel_acc
  eval_parallel_acc_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_parallel_acc
  core
  -lgtest -lgtest_main
)

add_executable(test_eval_serial_acc
  eval_serial_acc_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_serial_acc
  core
  -lgtest -lgtest_main
)

add_executable(test_eval_serial_speed
  eval_serial_speed_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_serial_speed
  core
  -lgtest -lgtest_main
)
