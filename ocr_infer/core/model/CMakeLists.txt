add_library(
  model
  detect/db.cc
  detect/db_pp.cc
  recognize/crnn.cc
)
target_link_libraries(
  model
  clipper
  -lnvinfer -lnvinfer_plugin -lcudart
)
