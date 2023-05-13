# "-fPIC" is necessary when building shared library which depend on static library or object.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_subdirectory(core)
add_subdirectory(engines)
add_subdirectory(eval)

aux_source_directory(api SRC_API)
add_library(ocr_infer SHARED
  ${SRC_API}
  $<TARGET_OBJECTS:engines>
)
target_link_libraries(ocr_infer
  core
)
