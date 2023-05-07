add_subdirectory(core)

add_executable(main test/main.cc)

target_link_libraries(
  main
  core
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -lglog
  -lpthread
)
