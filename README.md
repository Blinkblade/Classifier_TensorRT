# Classifier_TensorRT

## Overview
The complete process from generating Pytorch models to using TensorRT inference.

一个使用TensoRT实现图像分类的全流程, 过程为:pytorch pth ===> onnx ===> engine

旨在从一个整体到细节的视角展现如何从0构建整个推理流程

## 快速开始/Quick Start
### 环境配置/Environment setup

基础环境基于nvidia TensorRT docker容器: nvcr.io/nvidia/tensorrt:22.10-py3 构建, 配置如下:
-  Ubuntu 20.04 x86_64
-  CUDA 11.8
-  cuDNN 8.6.0.163
-  TensorRT 8.5 EA.
-  g++ 9.4.0
-  cmake 3.24.0
  
额外配置如下:
- PyTorch 2.0.1
- OpenCV 4.5.5

请根据torch和opencv官网自行安装.

此外,源码还在Jetson上进行了部署与测试,部署设备环境:
-  Jetson Orin Nano
-  Ubuntu 20.04 aarch64
-  CUDA 11.4
-  cuDNN 8.6.0.166
-  TensorRT 8.5.2.2
-  OpenCV 4.5.4
-  g++ 9.4.0
-  cmake 3.16.3
-  PyTorch 1.12.0
  
经测试无需特殊修改即可成功编译并运行。如果是本地安装的TensorRT,可能需要修改CMakeLists.txt的对应路径,详情在CMakeLists.txt中已经详细注解。

### 运行示例/Example of quick start commands
#### 克隆代码/clone the code
```bash
git clone https://github.com/Blinkblade/Classifier_TensorRT.git
cd Classifier_TensorRT
```
#### 创建文件夹/make dirs
```bash
mkdir build
mkdir -p models/onnx
mkdir -p models/engine
```
#### 导出onnx/Export_onnx
```bash
python scripts/export_onnx.py 
```
这会创建一个torchvision 版本的 resnet34 onnx.

##### 导出yolov8 onnx用于2D目标检测/Export onnx for 2D object detection based on yolov8
可以使用yolov8 onnx测试2D目标检测任务:

首先需要按照yolov8官方项目安装yolov8: https://github.com/ultralytics/ultralytics

然后, 为了保证cpp解码时访存方便,需要首先找到yolov8 head的输出位置,将输出的形状从 B C N转换为B N C

例如：

这个路径下: ultralytics/ultralytics/nn/modules/head.py

line 61 行 加入: 
```python
y = y.transpose(1,2)
```
然后运行export_yolov8.py脚本导出
```bash
python scripts/export_yolov8.py 
```
再将yolov8.onnx移动到models/onnx目录下
```bash
mv yolov8n.onnx ./models/onnx/
```
#### 使用CMAKE编译/Compile with cmake
首先需要将CMakeLists.txt中的tensorrt路径TENSORRT_INSTALL_DIR改为你的安装路径,不同的安装模式有不同写法,这点在CMakeLists.txt已经注明
```bash
cd build
cmake ..
make -j16
cd ..
```

#### 运行可执行程序/Run the executable
```bash
./main
```

## Features
- 从pytorch -> onnx -> tensorrt 的全流程
- 健壮且泛用的代码,轻松实现跨平台部署(在Jetson Orin Nano上进行了部署与测试)
- 大量注释涵括从整体到细节,保证和我一样的菜鸡也可以快速学习如何从0实现TensorRT推理
- 写了很多测试函数可以灵活地测试单个功能模块
- 所有功能封装到工厂类中,只需指定图像即可实现一键推理
- 使用cuda进行前处理加速,一次访存可以并行处理所有的前处理阶段
- 使用cmake实现直观高效的c++/cuda混合编译,再也不用为找不到路径/看不懂Makefile发愁了!
- note中记录了一些笔记,是构建项目时遇到的一些问题,如果有和我一样的小白可以看看避一些坑


## Acknowledgements
代码结构参考的是这个项目:

https://github.com/kalfazed/tensorrt_starter

本项目按照该项目思路重新编写,使用cmake构建并按照个人喜好进行了修改和优化(我认为的)

鄙人才疏学浅,若认为项目中有不完善,有疏漏或你认为可以改进的地方,可以随时和我联系!

我仍在不断学习中,热切希望和各位大佬探讨.

Feel free to contact me!

## TODO

- [x] 在Jetson Orin Nano上进行部署(经过测试,本项目无需特别修改可以直接在Jetson开发板上成功编译并运行。)
- [ ] 使用摄像头进行实时推理
- [x] 2D object detection
- [ ] 3d object detection
- [ ] multimodal 3d object detection
      
