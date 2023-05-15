add_executable(test_eval_speed
  eval_speed_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_speed
  core
  -lgtest -lgtest_main
)

add_executable(test_eval_acc
  eval_acc_test.cc
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(test_eval_acc
  core
  -lgtest -lgtest_main
)
