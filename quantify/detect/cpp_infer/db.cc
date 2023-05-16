#include "db.h"

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

class DetTrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO) printf("%s\n", msg);
  }
};

DetTrtLogger detLogger;

Db::Db(const std::string &model_path, int batch_size)
    : batch_size_(batch_size) {
  LOG(INFO) << "Loading detect model " << model_path;
  std::ifstream engine_file(model_path.c_str(),
                            std::ios::in | std::ios::binary);
  CHECK(engine_file.good()) << "Can't load detect model file";
  std::vector<char> model_stream;
  engine_file.seekg(0, engine_file.end);
  size_t model_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  model_stream.resize(model_size);
  engine_file.read(model_stream.data(), model_size);
  engine_file.close();
  LOG(INFO) << "Detect model size: "
            << static_cast<unsigned long long>(model_stream.size());

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(detLogger);
  CHECK(runtime) << "Detect runtime creation failed!";

  engine_ =
      runtime->deserializeCudaEngine(model_stream.data(), model_size, nullptr);
  CHECK(engine_) << "Detect engine deserialize failed!";

  context_ = engine_->createExecutionContext();
  CHECK(context_) << "Create detection context failed!";

  runtime->destroy();

  input_index_ = engine_->getBindingIndex("inputs");
  output_index_ = engine_->getBindingIndex("outputs");
  nvinfer1::Dims input_dim =
      engine_->getBindingDimensions(engine_->getBindingIndex("inputs"));
  context_->setOptimizationProfile(0);
  context_->setBindingDimensions(
      input_index_, nvinfer1::Dims4(batch_size_, input_dim.d[1], input_dim.d[2],
                                    input_dim.d[3]));
  host_buffers_.resize(2);
  gpu_buffers_.resize(2);
  MYCHECK(cudaStreamCreate(&stream_));
  input_size_ = sizeof(float) * batch_size_ * 3 * 512 * 512;
  output_size_ = sizeof(float) * batch_size_ * 512 * 512;
  CHECK(HostMalloc(&host_buffers_[input_index_], input_size_) &&
        HostMalloc(&host_buffers_[output_index_], output_size_))
      << "Can't malloc host memory!";
  CHECK(GpuMalloc(&gpu_buffers_[input_index_], input_size_) &&
        GpuMalloc(&gpu_buffers_[output_index_], output_size_))
      << "Can't malloc gpu memory!";
  input_array_ = static_cast<float *>(host_buffers_[input_index_]);
  output_array_ = static_cast<float *>(host_buffers_[output_index_]);
  LOG(INFO) << "Detect model is loaded.";
}

Db::~Db() {
  MYCHECK(cudaStreamSynchronize(stream_));
  MYCHECK(cudaStreamDestroy(stream_));
  for (void *&p : host_buffers_) {
    cudaFreeHost(p);
  }
  for (void *&p : gpu_buffers_) {
    cudaFree(p);
  }
  if (context_) {
    context_->destroy();
  }
  if (engine_) {
    engine_->destroy();
  }
}

void Db::Forward(const std::vector<cv::Mat> &in, std::vector<cv::Mat> *out) {
  Inference(in);
  MYCHECK(cudaStreamSynchronize(stream_));
  for (size_t h = 0, vol = 512 * 512; h < in.size(); h++) {
    cv::Mat res(cv::Size(512, 512), CV_32FC1);
    memcpy((float *)res.ptr<float>(0), output_array_ + h * vol,
           sizeof(float) * vol);
    out->emplace_back(res);
  }
}

bool Db::HostMalloc(void **ptr, size_t size) {
  return cudaMallocHost(ptr, size) == cudaSuccess;
}

bool Db::GpuMalloc(void **ptr, size_t size) {
  return cudaMalloc(ptr, size) == cudaSuccess;
}

// TODO: 待优化，省去循环
void Db::Inference(const std::vector<cv::Mat> &imgs) {
  for (size_t i = 0, vol = 3 * 512 * 512; i < imgs.size(); i++) {
    VLOG(1) << "i = " << i;
    VLOG(1) << "img dim: " << imgs[i].dims;
    VLOG(1) << "img channels: " << imgs[i].channels();
    VLOG(1) << "img size: " << imgs[i].size;
    if (imgs[i].isContinuous()) {
      VLOG(1) << "continue";
      memcpy(input_array_ + i * vol, (float *)imgs[i].ptr<float>(0),
             sizeof(float) * vol);
    } else {
      cv::Mat ii = imgs[i].clone();
      memcpy(input_array_ + i * vol, (float *)ii.ptr<float>(0),
             sizeof(float) * vol);
    }
  }
  MYCHECK(cudaMemcpyAsync(gpu_buffers_[input_index_],
                          host_buffers_[input_index_], input_size_,
                          cudaMemcpyHostToDevice, stream_));
  CHECK(context_->enqueueV2(gpu_buffers_.data(), stream_, nullptr))
      << "Detect inference failed.";
  MYCHECK(cudaMemcpyAsync(host_buffers_[output_index_],
                          gpu_buffers_[output_index_], output_size_,
                          cudaMemcpyDeviceToHost, stream_));
}
