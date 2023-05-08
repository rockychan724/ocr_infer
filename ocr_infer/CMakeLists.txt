add_subdirectory(core)
add_subdirectory(engines)

add_executable(main test/main.cc)
target_link_libraries(
  main
  core
)

# aux_source_directory(api SRC_API)
# add_library(ocrinfer SHARED ${SRC_API})
# target_link_libraries(
#   ocrinfer
#   engines
# )

file(GLOB_RECURSE src_list ./*.cc ./*.cpp)
add_library(ocrinfer SHARED ${src_list})
target_link_libraries(
  ocrinfer
  -lnvinfer -lnvinfer_plugin -lcudart
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -ltbb
  -lglog
  -lpthread
  -lgtest -lgtest_main
)
