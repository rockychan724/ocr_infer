cmake_minimum_required(VERSION 3.16)
project(untitled)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_DEBUG_POSTFIX "_d")

option(SAVE_CLIPS "Wether to save recognition input images." OFF)
option(WRITE_LOG_TO_FILE "Wether to write log info to file." OFF)

if(SAVE_CLIPS)
    add_definitions(-DSAVE_CLIPS)
endif()
if(WRITE_LOG_TO_FILE)
    add_definitions(-DWRITE_LOG_TO_FILE)
endif()

enable_testing()

set(CMAKE_CXX_STANDARD 17)

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/local/include
    /usr/include/opencv4
    /usr/local/cuda/include
    # /usr/local/tensorrt/include
)

link_directories(
    /usr/local/lib
    # /usr/local/opencv4/lib
    /usr/local/cuda/lib64
    # /usr/local/tensorrt/lib
)

add_subdirectory(ocr_infer)
