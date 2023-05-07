cmake_minimum_required(VERSION 3.16)
project(untitled)

enable_testing()

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/local/include
    /usr/local/opencv4/include/opencv4
    /usr/local/cuda/include
    /usr/local/tensorrt/include
)

link_directories(
    /usr/local/lib
    /usr/local/opencv4/lib
    /usr/local/cuda/lib64
    /usr/local/tensorrt/lib
)

add_subdirectory(ocr_infer)
