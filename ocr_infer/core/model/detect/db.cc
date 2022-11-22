#include "ocr_infer/core/model/detect/db.h"

#include <fstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "glog/logging.h"

#define MYCHECK(status)                                  \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

Db::Db(const std::string &model_path, int batch_size) : batchsize(batch_size) {
  LOG(INFO) << "Loading detect model " << model_path;
  std::ifstream engine_file(model_path.c_str(), std::ios::in | std::ios::binary);
  CHECK(engine_file.good()) << "Can't load detect model file";
  std::vector<char> model_stream;
  engine_file.seekg(0, engine_file.end);
  size_t model_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  model_stream.resize(model_size);
  engine_file.read(model_stream.data(), model_size);
  engine_file.close();
  LOG(INFO) << "Detection model size: " << static_cast<unsigned long long>(model_stream.size());

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(detLogger);
  CHECK(runtime) << "Detection runtime creation failed!";

  engine = runtime->deserializeCudaEngine(model_stream.data(), model_size, nullptr);
  CHECK(engine) << "Detection engine deserialize failed!";

  context = engine->createExecutionContext();
  CHECK(context) << "Create detection context failed!";

  runtime->destroy();
  nvinfer1::Dims input_dim = engine->getBindingDimensions(engine->getBindingIndex("inputs"));
  context->setOptimizationProfile(0);
  context->setBindingDimensions(
      engine->getBindingIndex("inputs"),
      nvinfer1::Dims4(batchsize, input_dim.d[1], input_dim.d[2], input_dim.d[3]));
  hostbuffers.resize(2);
  gpubuffers.resize(2);
  MYCHECK(cudaStreamCreate(&stream));
  inputindex = engine->getBindingIndex("inputs");
  outputindex = engine->getBindingIndex("outputs");
  inputsize = sizeof(float) * batchsize * 3 * 512 * 512;
  outputsize = sizeof(float) * batchsize * 512 * 512;
  CHECK(HostMalloc(&hostbuffers[inputindex], inputsize) &&
        HostMalloc(&hostbuffers[outputindex], outputsize))
      << "Can't malloc host memory!";
  CHECK(GpuMalloc(&gpubuffers[inputindex], inputsize) &&
        GpuMalloc(&gpubuffers[outputindex], outputsize))
      << "Can't malloc gpu memory!";
  inputarray = static_cast<float *>(hostbuffers[inputindex]);
  outputarray = static_cast<float *>(hostbuffers[outputindex]);
  LOG(INFO) << "Detection model is loaded.";
}

Db::~Db() {
  MYCHECK(cudaStreamSynchronize(stream));
  MYCHECK(cudaStreamDestroy(stream));
  for (void *&p : hostbuffers) cudaFreeHost(p);
  for (void *&p : gpubuffers) cudaFree(p);
  if (context) context->destroy();
  if (engine) engine->destroy();
}

void Db::Forward(const std::vector<cv::Mat> &in, std::vector<cv::Mat> *out) {
  Inference(in);
  MYCHECK(cudaStreamSynchronize(stream));
  for (size_t h = 0, vol = 512 * 512; h < in.size(); h++) {
    cv::Mat res(cv::Size(512, 512), CV_32FC1);
    memcpy((float *)res.ptr<float>(0), outputarray + h * vol, sizeof(float) * vol);
    out->emplace_back(res);
  }
}

bool Db::HostMalloc(void **ptr, size_t size) { return cudaMallocHost(ptr, size) == cudaSuccess; }

bool Db::GpuMalloc(void **ptr, size_t size) { return cudaMalloc(ptr, size) == cudaSuccess; }

// TODO: 待优化，省去循环
void Db::Inference(const std::vector<cv::Mat> &imgs) {
  for (size_t i = 0, vol = 3 * 512 * 512; i < imgs.size(); i++) {
    VLOG(1) << "i = " << i;
    VLOG(1) << "img dim: " << imgs[i].dims;
    VLOG(1) << "img channels: " << imgs[i].channels();
    VLOG(1) << "img size: " << imgs[i].size;
    if (imgs[i].isContinuous()) {
      VLOG(1) << "continue";
      memcpy(inputarray + i * vol, (float *)imgs[i].ptr<float>(0), sizeof(float) * vol);
    } else {
      cv::Mat ii = imgs[i].clone();
      memcpy(inputarray + i * vol, (float *)ii.ptr<float>(0), sizeof(float) * vol);
    }
  }
  MYCHECK(cudaMemcpyAsync(gpubuffers[inputindex], hostbuffers[inputindex], inputsize,
                          cudaMemcpyHostToDevice, stream));
  CHECK(context->enqueueV2(gpubuffers.data(), stream, nullptr)) << "Detect inference failed.";
  MYCHECK(cudaMemcpyAsync(hostbuffers[outputindex], gpubuffers[outputindex], outputsize,
                          cudaMemcpyDeviceToHost, stream));
}
