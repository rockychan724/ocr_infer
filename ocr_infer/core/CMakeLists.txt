add_subdirectory(match)
add_subdirectory(model)
add_subdirectory(node_core)
add_subdirectory(pipeline)
add_subdirectory(util)

# add static library
add_library(core
  $<TARGET_OBJECTS:match>
  $<TARGET_OBJECTS:model>
  $<TARGET_OBJECTS:node_core>
  $<TARGET_OBJECTS:pipeline>
  $<TARGET_OBJECTS:file_handle>
  $<TARGET_OBJECTS:clipper>
)
target_link_libraries(core
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
  -lnvinfer -lnvinfer_plugin -lcudart
  -ltbb
  -lglog
)
