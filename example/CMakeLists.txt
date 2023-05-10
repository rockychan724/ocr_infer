cmake_minimum_required(VERSION 3.16)

project(ocr_demo)

include_directories(${PROJECT_SOURCE_DIR}/ocr_infer/include)

link_directories(
  ${PROJECT_SOURCE_DIR}/ocr_infer/lib
)

add_executable(ocr_demo src/main.cc)
target_link_libraries(
  ocr_demo
  -locrinfer
)
