cmake_minimum_required(VERSION 3.16)
project(untitled)

enable_testing()

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/include/opencv4
    /usr/local/cuda/include
)

link_directories(
    /usr/local/cuda/lib64
)

add_subdirectory(ocr_infer)
