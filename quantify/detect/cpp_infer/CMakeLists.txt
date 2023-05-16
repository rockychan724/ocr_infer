cmake_minimum_required(VERSION 3.16)
project(detect_infer)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party
    /usr/include/opencv4
    /usr/local/cuda/include
)

link_directories(
    /usr/local/cuda/lib64
)

add_executable(detect_infer main.cc db.cc db_pp.cc third_party/clipper.cpp)
target_link_libraries(
    detect_infer
    -lnvinfer
    -lnvinfer_plugin
    -lcudart
    -lopencv_core
    -lopencv_imgcodecs
    -lopencv_imgproc
    -lglog
)
