cmake_minimum_required(VERSION 3.16)
project(rec_infer)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/include/opencv4
    /usr/local/cuda/include
)

link_directories(
    /usr/local/cuda/lib64
)

add_executable(rec_infer main.cc crnn.cc)
target_link_libraries(rec_infer
    -lnvinfer
    -lnvinfer_plugin
    -lcudart
    -lopencv_core
    -lopencv_imgcodecs
    -lopencv_imgproc
    -lglog
)
