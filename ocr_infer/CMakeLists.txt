
add_subdirectory(core)

# file(GLOB node_core core/node_core/*.cc)
file(GLOB_RECURSE core core/*.cc core/*.cpp)
message("*** core: ${core}")

add_executable(main test/main.cc ${core})

target_link_libraries(
  main
  -lnvinfer -lnvinfer_plugin -lcudart
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -ltbb
  -lglog
)
