#ifndef OCR_INFER_CORE_MODEL_RECOGNIZE_CRNN_H_
#define OCR_INFER_CORE_MODEL_RECOGNIZE_CRNN_H_

#include <string>
#include <vector>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"

class Crnn {
 public:
  Crnn(const std::string &model_path, const std::string &dict_path, int batch_size);

  ~Crnn();

  void Forward(const std::vector<cv::Mat> &in, std::vector<std::string> *out);

 private:
  std::vector<std::string> dict_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  cudaStream_t stream_;
  std::vector<void *> host_buffers_, gpu_buffers_;
  int input_index_, output_index_, batch_size_;
  size_t input_size_, output_size_;
  float *input_array_;
  int32_t *output_array_;

  void LoadDictionary(const std::string &dict_path);

  bool HostMalloc(void **ptr, size_t size);

  bool GpuMalloc(void **ptr, size_t size);

  void Inference(const std::vector<cv::Mat> &imgs);

  std::string Decode(const std::vector<int> &max_index);
};

#endif  // OCR_INFER_CORE_MODEL_RECOGNIZE_CRNN_H_
