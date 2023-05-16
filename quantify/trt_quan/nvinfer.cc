#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
using namespace std;
using namespace nvonnxparser;
using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    if (severity != Severity::kINFO) printf("%s\n", msg);
  }
} gLogger;

void printUsage(char *programPath) {
  printf(
      "Usage: %s [path_to_onnx] [max_batch_size] [path_to_engine]\n\
[path_to_onnx]: Input onnx file path.\n\
[max_batch_size]: Max batch when inference.\n\
[path_to_engine]: Output engine file path.\n",
      programPath);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    printf(
        "An util to convert an onnx model to FP16 tensorrt engine.\n\
Note: The INT8 part has been removed for simplicity.\n");
    printf("onnx version: %d\n", getNvOnnxParserVersion());
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  } else if (argc != 4) {
    printf("Bad arguments.\n");
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
  const char *onnx_path = argv[1];
  const int max_batch = atoi(argv[2]);
  const char *trt_path = argv[3];
  printf("onnx path: %s\n", onnx_path);
  IBuilder *builder = createInferBuilder(gLogger);
  if (builder->platformHasFastFp16())
    printf("Faster FP16 supported in this platform.\n");
  else
    printf(
        "Faster FP16 not supported. TensorRT will fallback to FP32 or TF32.\n");
  if (builder->platformHasTf32())
    printf("Faster TF32 supported in this platform.\n");
  else
    printf(
        "Faster TF32 not supported in this platform. TensorRT will fallback to "
        "FP32.\n");
  printf("Number of DLA cores: %d\n", builder->getNbDLACores());
  printf("Enter a character before beginning...\n");
  getchar();
  INetworkDefinition *network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  IParser *parser = createParser(*network, gLogger);
  if (!parser) {
    printf("create parser failed.\n");
    return EXIT_FAILURE;
  }
  if (!(parser->parseFromFile(onnx_path,
                              static_cast<int>(ILogger::Severity::kWARNING)))) {
    printf("parse onnx failed.\n");
    return EXIT_FAILURE;
  }
  IBuilderConfig *config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 30);
  config->setFlag(BuilderFlag::kFP16);
  if (builder->getNbDLACores() > 0) {
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setDLACore(0);
  } else
    config->setDefaultDeviceType(DeviceType::kGPU);
  IOptimizationProfile *profile = builder->createOptimizationProfile();
  ITensor *input_node = network->getInput(0);
  input_node->setAllowedFormats(1U
                                << static_cast<uint32_t>(TensorFormat::kHWC));
  Dims input_dim = input_node->getDimensions();
  profile->setDimensions(
      input_node->getName(), OptProfileSelector::kMIN,
      Dims4(1, input_dim.d[1], input_dim.d[2], input_dim.d[3]));
  profile->setDimensions(
      input_node->getName(), OptProfileSelector::kMAX,
      Dims4(max_batch, input_dim.d[1], input_dim.d[2], input_dim.d[3]));
  profile->setDimensions(
      input_node->getName(), OptProfileSelector::kOPT,
      Dims4(max_batch / 2 + 1, input_dim.d[1], input_dim.d[2], input_dim.d[3]));
  config->addOptimizationProfile(profile);
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) {
    printf("build engine failed.\n");
    return EXIT_FAILURE;
  }
  printf("build engine successfully.\n");
  IHostMemory *serialize = engine->serialize();
  if (!serialize) {
    printf("engine serialize failed.\n");
    return EXIT_FAILURE;
  }
  ofstream file(trt_path, ios::out | ios::binary);
  file.write(static_cast<char *>(serialize->data()), serialize->size());
  printf("save to %s\n", trt_path);
  file.close();
  return EXIT_SUCCESS;
}