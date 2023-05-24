# ocr_infer

## 介绍

- 简介：OCR 离线推理引擎
- 功能：处理大规模复杂网络图像，并从中提取关键词，该系统包括文字检测、文字识别、关键词匹配三大功能
- 技术： 
   - 使用 TensorRT 量化、加速；
   - 使用生产者-消费者模型将系统内部模块并行化，以提高系统的吞吐量；
   - 使用 DB 作为文字检测模型、CRNN 作为文字识别模型；
   - 使用 AC 自动机和模糊匹配来查找关键词。
## 软件架构
系统分成文字检测、检测后处理、缓存、文字识别、识别后处理、关键词匹配六个模块。
## 使用教程

1. 下载代码
```shell
git clone https://github.com/ruyueshi/ocr_infer.git
```

2. 下载模型和数据（联系ruyueshi@qq.com）

3. 配置开发环境（两种方式，选其中一个即可）：

- 根据 Dockerfile 来构建开发镜像
```shell
cd ocr_infer
docker build -t my_ocr_infer:v1 .
```

- 直接拉去已经配置好的开发环境
```shell
docker pull ruyueshi/ocr_infer:v1-cuda11.1.1-cudnn8-tensorrt7
```

4. 编译：
```shell
cd ocr_infer
cmake -S . -B build
cmake --build build -j8  # 默认会构建所有目标
```

5. 运行：
- 测试系统精度
```shell
# 运行测试精度的单元测试
./build/ocr_infer/eval/test_eval_parallel_acc

# 运行评测脚本
cd eval_acc
python3 eval_end2end_v1.py ../testdata/e2e/gt/ ../output/rec_output/
```

- 测试系统速度
```shell
# 运行测试速度的单元测试
./build/ocr_infer/eval/test_eval_parallel_speed

# 查看运行结果
cat ./output/speed.txt
```
