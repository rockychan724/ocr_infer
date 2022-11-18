cmake_minimum_required(VERSION 3.16)
project(untitled)

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/include/opencv4
)

add_subdirectory(ocr_infer)
