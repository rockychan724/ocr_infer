# build TensorRT quantify tool
./build.sh

# quantify onnx model into trt offline engine

## detect with mobilenet
### on 3090
CUDA_VISIBLE_DEVICES=1 ./nvinfer ../weights/onnx/db_mobilenet.onnx 50 ../weights/trt_engine/3090/db_mobilenet_50.fp16
### on 2080Ti
CUDA_VISIBLE_DEVICES=2 ./nvinfer ../weights/onnx/db_mobilenet.onnx 50 ../weights/trt_engine/2080Ti/db_mobilenet_50.fp16

## detect with resnet
### on 3090
CUDA_VISIBLE_DEVICES=1 ./nvinfer ../weights/onnx/db_resnet.onnx 50 ../weights/trt_engine/3090/db_resnet_50.fp16
### on 2080Ti
CUDA_VISIBLE_DEVICES=2 ./nvinfer ../weights/onnx/db_resnet.onnx 50 ../weights/trt_engine/2080Ti/db_resnet_50.fp16

## recognize
### on 3090
CUDA_VISIBLE_DEVICES=1 ./nvinfer ../weights/onnx/crnn.onnx 100 ../weights/trt_engine/3090/crnn_100.fp16
### on 2080Ti
CUDA_VISIBLE_DEVICES=2 ./nvinfer ../weights/onnx/crnn.onnx 100 ../weights/trt_engine/2080Ti/crnn_100.fp16
