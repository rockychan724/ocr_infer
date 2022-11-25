add_subdirectory(match)
add_subdirectory(model)
add_subdirectory(util)

aux_source_directory(node_core SRC_NODE_CORE)
aux_source_directory(pipeline SRC_PIPELINE)
message("*** node_core: ${SRC_NODE_CORE}")
message("*** pipeline: ${SRC_PIPELINE}")

add_library(core
  ${SRC_NODE_CORE}
  ${SRC_PIPELINE}
)
target_link_libraries(
  core
  match model
  -ltbb
)
