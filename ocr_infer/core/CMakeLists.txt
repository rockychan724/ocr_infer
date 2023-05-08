add_subdirectory(match)
add_subdirectory(model)
add_subdirectory(util)

aux_source_directory(node_core SRC_NODE_CORE)
aux_source_directory(pipeline SRC_PIPELINE)

add_library(core
  ${SRC_NODE_CORE}
  ${SRC_PIPELINE}
)
target_link_libraries(
  core
  match model
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -ltbb
  -lglog
  -lpthread
)
