add_subdirectory(core)
add_subdirectory(engines)

add_executable(main eval/main.cc)
target_link_libraries(
  main
  core
)

# aux_source_directory(api SRC_API)
# add_library(ocr_infer SHARED ${SRC_API})
# target_link_libraries(
#   ocr_infer
#   engines
# )

# TODO: 应当避免重复编译
file(GLOB_RECURSE src_list ./*.cc ./*.cpp)
add_library(ocr_infer SHARED ${src_list})
target_link_libraries(
  ocr_infer
  -lnvinfer -lnvinfer_plugin -lcudart
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -ltbb
  -lglog
  -lpthread
  -lgtest -lgtest_main
)
