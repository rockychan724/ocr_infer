cmake_minimum_required(VERSION 3.16)

project(ocr_demo)

include_directories(
  ${PROJECT_SOURCE_DIR}/ocr_infer/include
  /usr/local/opencv4/include/opencv4
)

link_directories(
  ${PROJECT_SOURCE_DIR}/ocr_infer/lib
  /usr/local/opencv4/lib
)

add_executable(ocr_demo src/main.cc)
target_link_libraries(
  ocr_demo
  -locr_infer
)
