#ifndef OCR_INFER_CORE_MODEL_DETECT_DB_H_
#define OCR_INFER_CORE_MODEL_DETECT_DB_H_

#include <string>
#include <vector>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"

/**
 * @brief DB model in text detection: https://arxiv.org/abs/1911.08947
 */
class Db {
 public:
  Db(const std::string &model_path, int batch_size);

  ~Db();

  void Forward(const std::vector<cv::Mat> &in, std::vector<cv::Mat> *out);

 private:
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  cudaStream_t stream_;
  std::vector<void *> host_buffers_, gpu_buffers_;
  int input_index_, output_index_, batch_size_;
  size_t input_size_, output_size_;
  float *input_array_, *output_array_;

  bool HostMalloc(void **ptr, size_t size);

  bool GpuMalloc(void **ptr, size_t size);

  void Inference(const std::vector<cv::Mat> &imgs);
};

#endif  // OCR_INFER_CORE_MODEL_DETECT_DB_H_
