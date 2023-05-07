#include "ocr_infer/core/model/recognize/crnn.h"

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

class RecTrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO) printf("%s\n", msg);
  }
};
RecTrtLogger recLogger;

Crnn::Crnn(const std::string &model_path, const std::string &dict_path,
           int batch_size)
    : batch_size_(batch_size) {
  LOG(INFO) << "Loading recognize model " << model_path;

  LoadDictionary(dict_path);

  std::ifstream engine_file(model_path.c_str(),
                            std::ios::in | std::ios::binary);
  CHECK(engine_file.good()) << "Can't load recognize model file";
  std::vector<char> model_stream;
  engine_file.seekg(0, engine_file.end);
  size_t model_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  model_stream.resize(model_size);
  engine_file.read(model_stream.data(), model_size);
  engine_file.close();
  LOG(INFO) << "Recognize model size: "
            << static_cast<unsigned long long>(model_stream.size());

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(recLogger);
  CHECK(runtime) << "Recognize runtime creation failed!";

  engine_ =
      runtime->deserializeCudaEngine(model_stream.data(), model_size, nullptr);
  CHECK(engine_) << "Recognize engine deserialize failed!";

  context_ = engine_->createExecutionContext();
  CHECK(context_) << "Create Recognize context failed!";

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
  input_size_ = sizeof(float) * batch_size_ * 480 * 48;
  output_size_ = sizeof(int32_t) * batch_size_ * 60;
  CHECK(HostMalloc(&host_buffers_[input_index_], input_size_) &&
        HostMalloc(&host_buffers_[output_index_], output_size_))
      << "Can't malloc host memory!";
  CHECK(GpuMalloc(&gpu_buffers_[input_index_], input_size_) &&
        GpuMalloc(&gpu_buffers_[output_index_], output_size_))
      << "Can't malloc gpu memory!";
  input_array_ = static_cast<float *>(host_buffers_[input_index_]);
  output_array_ = static_cast<int32_t *>(host_buffers_[output_index_]);
  LOG(INFO) << "Recognize model is loaded.";
}

Crnn::~Crnn() {
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

void Crnn::Forward(const std::vector<cv::Mat> &in,
                   std::vector<std::string> *out) {
  Inference(in);
  MYCHECK(cudaStreamSynchronize(stream_));
  for (int i = 0; i < in.size(); i++) {
    std::vector<int> max_indexes(output_array_ + i * 60,
                                 output_array_ + (i + 1) * 60);
    out->emplace_back(Decode(max_indexes));
  }
}

void Crnn::LoadDictionary(const std::string &dict_path) {
  LOG(INFO) << "Reading dict file...";
  std::ifstream f(dict_path.c_str());
  CHECK(f.is_open()) << "Can't open dict file!";

  std::string line;
  dict_.emplace_back("blank");
  while (std::getline(f, line)) {
    dict_.emplace_back(line);
  }
  f.close();
  dict_.emplace_back("UNKNOWN");
  LOG(INFO) << "Total word number in dict is " << dict_.size();
}

bool Crnn::HostMalloc(void **ptr, size_t size) {
  return cudaMallocHost(ptr, size) == cudaSuccess;
}

bool Crnn::GpuMalloc(void **ptr, size_t size) {
  return cudaMalloc(ptr, size) == cudaSuccess;
}

void Crnn::Inference(const std::vector<cv::Mat> &imgs) {
  for (size_t i = 0, vol = 48 * 480; i < imgs.size(); i++) {
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
      << "Recognize inference failed.";
  MYCHECK(cudaMemcpyAsync(host_buffers_[output_index_],
                          gpu_buffers_[output_index_], output_size_,
                          cudaMemcpyDeviceToHost, stream_));
}

std::string Crnn::Decode(const std::vector<int> &max_index) {
  std::string res;
  for (int i = 0; i < max_index.size(); i++) {
    if (max_index[i] <= 0 ||
        max_index[i] == max_index[i - 1])  // TODO: 去掉UNKNOWN
      continue;
    res += dict_[max_index[i]];
  }
  return res;
}
