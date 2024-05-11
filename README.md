# Classifier_TensorRT

## Overview
The complete process from generating Pytorch models to using TensorRT inference.

一个使用TensoRT实现图像分类的全流程, 过程为:pytorch pth ===> onnx ===> engine

旨在从一个新手的角度展现如何从0构建整个流程

## Quick Start
### Environment setup

基础环境基于nvidia TensorRT docker容器: nvcr.io/nvidia/tensorrt:22.10-py3 构建, 配置如下:
-  Ubuntu 20.04
-  CUDA 11.8.0
-  cuDNN 8.6.0.163
-  TensorRT 8.5 EA.
-  g++ 9.4.0
-  cmake 3.24.0
额外配置如下:
- PyTorch 2.0.1
- OpenCV 4.5.5
请根据torch和opencv官网自行安装.
### Example of quick start commands
#### clone the code
```bash
git clone https://github.com/yourusername/Classifier_TensorRT.git
cd Classifier_TensorRT
```
#### 导出onnx
```bash
python scripts/export_onnx.py 
```
#### Compile with cmake
```bash
cd build
cmake ..
make -j16
cd ..
```

#### Run the executable
```bash
./main
```

## Features
- 从pytorch -> onnx -> tensorrt 的全流程
- 大量注释涵括从整体到细节,保证和我一样的菜鸡也可以快速学习如何从0实现TensorRT推理
- 写了很多测试函数可以灵活地测试单个功能模块
- python,c++,cuda多语言
- 使用cmake实现直观高效的c++/cuda混合编译,再也不用为找不到路径/看不懂Makefile发愁了!


## Acknowledgements
代码结构参考的是这个项目:

https://github.com/kalfazed/tensorrt_starter

本项目按照该项目思路重新编写,使用cmake构建并按照个人喜好进行了修改和优化(我认为的)

鄙人才疏学浅,若认为项目中有不完善,有疏漏或你认为可以改进的地方,可以随时和我联系!

我仍在不断学习中,热切希望和各位大佬探讨.

Feel free to contact me!

## TODO

- [ ] 在Jetson Orin Nano上进行部署
- [ ] 使用摄像头进行实时推理
- [ ] 2D object detection
- [ ] 3d object detection
- [ ] multimodal 3d object detection
      



