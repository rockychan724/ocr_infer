#ifndef OCR_INFER_CORE_MODEL_DETECT_DB_H_
#define OCR_INFER_CORE_MODEL_DETECT_DB_H_

#include <string>
#include <vector>

#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "ocr_infer/core/model/model_base.h"
#include "opencv2/opencv.hpp"

class DetTrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity != Severity::kINFO) printf("%s\n", msg);
  }
};

static DetTrtLogger detLogger;

/**
 * @brief DB model in text detection: https://arxiv.org/abs/1911.08947
 */
class Db {
 public:
  Db(const std::string &model_path, int batch_size);

  ~Db();

  void Forward(const std::vector<cv::Mat> &in, std::vector<cv::Mat> *out);

 private:
  nvinfer1::ICudaEngine *engine;
  nvinfer1::IExecutionContext *context;
  cudaStream_t stream;
  std::vector<void *> hostbuffers, gpubuffers;
  int inputindex, outputindex, batchsize;
  size_t inputsize, outputsize;
  float *inputarray, *outputarray;

  bool HostMalloc(void **ptr, size_t size);

  bool GpuMalloc(void **ptr, size_t size);

  void Inference(const std::vector<cv::Mat> &imgs);
};

#endif  // OCR_INFER_CORE_MODEL_DETECT_DB_H_
