# ocr_infer

## 介绍
- 简介：ocr_infer 是一款 OCR 离线推理引擎，能够处理大规模复杂新闻场景的网络图像，从图像中提取关键词，该系统包括文字检测、文字识别、关键词匹配三大功能。
- 技术： 
   - 使用 TensorRT 量化、加速；
   - 使用生产者-消费者模型将系统内部模块并行化，以提高系统的吞吐量；
   - 使用 DB 作为文字检测模型、CRNN 作为文字识别模型；
   - 使用 AC 自动机和模糊匹配来查找关键词。

## 软件架构
为了提高系统的吞吐量，使用生产者-消费者模型将系统内部各个模块并行化，并根据流水线原理，尽可能细分节点以提高处理速度，最后将系统细分成文字检测、检测后处理、文本行裁剪、缓存队列、文字识别、收集识别结果、关键词匹配这七个节点，后续可以根据具体的任务需求来进行扩展和维护。

## 运行步骤
1. 下载模型和数据（联系ruyueshi@qq.com）

2. 配置开发环境（为简化开发者配置环境的过程，现已将开发环境打包到镜像中，下面有两种获取开发镜像的方式，选其中一个即可，推荐选第二种）：
   1. 根据 Dockerfile 来构建开发镜像：
```shell
cd ocr_infer
docker build -t my_ocr_infer:v1 .
```

   2. 直接拉取已经配置好的开发镜像：
```shell
docker pull ruyueshi/ocr_infer:v1-cuda11.1.1-cudnn8-tensorrt7
```

4. 使用容器开发，根据开发镜像，生成容器：
```shell
docker run -it --gpus all -w /home --name my_ocr_infer  ruyueshi/ocr_infer:v1-ubuntu20-cuda11.1.1-cudnn8-tensorrt7 bash
```

5. 下载代码：
```shell
# 在容器中
cd /home
git clone https://github.com/ruyueshi/ocr_infer.git
```

6. 将下载好的模型和数据解压到 /home/ocr_infer 下：
```shell
# 在物理机中将数据拷贝到开发容器中
docker cp ocr_infer_data.zip my_ocr_infer:/home

# 在容器中解压数据
cd /home
unzip ocr_infer_data.zip
mv ocr_infer_data/data ocr_infer_data/testdata /home/ocr_infer && \
mv ocr_infer_data/quantify/weights /home/ocr_infer/quantify
```

7. 编译：
```shell
cd ocr_infer
cmake -S . -B build
cmake --build build -j8  # 默认会构建所有目标
```

8. 运行：
   1. 测试系统精度
```shell
# 运行测试精度的单测
./build/ocr_infer/eval/test_eval_parallel_acc

# 运行评测脚本
cd eval_acc
python3 eval_end2end_v1.py ../testdata/e2e/gt/ ../output/rec_output/
```

   2. 测试系统速度
```shell
# 运行测试速度的单测
./build/ocr_infer/eval/test_eval_parallel_speed

# 查看运行结果
cat ./output/speed.txt
```
