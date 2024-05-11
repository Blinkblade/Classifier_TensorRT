cmake_minimum_required(VERSION 3.10)

# 编译cuda程序需要设置语言选项
project(trt_classification)
enable_language(CUDA)#激活CUDA语言支持

# 设置CUDA OPENC TENSORRT目录
# set(CUDA_INCLUDE_DIRS "/usr/local/cuda-11.8")
# set(OPENCV_INSTALL_DIR "/usr/local/include/opencv4")
# 由于cuda和opencv都可以使用findpackage,无需特意指定路径
# 容器路径
set(TENSORRT_INSTALL_DIR "/opt/tensorrt")

# # 本地路径
# set(TENSORRT_INSTALL_DIR "/home/${usrname}/packages/TensorRT-8.5.3.1")
# # 第三方安装的trt需要手动链接lib,因为安装目录一般不在标准路径下,编译器找不到对应库
# link_directories(${TENSORRT_INSTALL_DIR}/lib)



# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 设置调试和优化
# 设置不同的构建类型的编译标志,g代表可调试,o3代表编译器自动优化代码
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# 同样，为 CUDA 设置不同的编译选项
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")  
# 根据你的GPU架构调整,3060 是 8.6
set(CUDA_ARCHITECTURES "8.6")


# 使用find package设置可用模块
# CUDA和OPENCV的需求可以通过find packge自动搜寻
# REQUIRED表示这个库是必须的
# 写了一些打印find的变量值, 相关变量可以再cmake或模块官方文档中查找
# 例如在cmake官方文档中查找findCUDA
find_package(CUDA REQUIRED)
message("==========CUDA find pacage==============")
message("CUDA found: ${CUDA_FOUND}")
message("CUDA Version: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
message("CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message("CUDA Cuda RT library: ${CUDA_LIBRARIES}")
find_package(OpenCV REQUIRED)
message("==========OpenCV find package===========")
message("OpenCV found: ${OpenCV_FOUND}")
message("OpenCV libraries: ${OpenCV_LIBS}")
message("OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")


# 头文件目录
# 本地头文件
include_directories(include)
include_directories(src)
# include_directories(cuda)
# 直接使用find cuda的路径
include_directories(${CUDA_INCLUDE_DIRS})
# OPENCV头文件,就是安装路径
# opencv使用find即可,不用自己找路径了
# include_directories(${OPENCV_INSTALL_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
# TENSORRT头文件
include_directories(${TENSORRT_INSTALL_DIR}/include)



# 自动寻找src下的所有cpp
file(GLOB_RECURSE SOURCES "src/*.cpp")
# 自动寻找CUDA下的所有cu
file(GLOB_RECURSE CUDA_SOURCES "cuda/*.cu")



# 设置可执行文件输出的目录到项目的顶层目录, 主程序会被生成到与cmakelist同级
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
message(("设置可执行文件目录为: ${CMAKE_SOURCE_DIR}")) 
add_executable(main main.cpp ${SOURCES} ${CUDA_SOURCES})


# 链接外部库
target_link_libraries(main
    ${CUDA_LIBRARIES}  # find package
    ${OpenCV_LIBS}      # find package自动赋值
    nvinfer            # TensorRT主库
    nvonnxparser       # 解析库
    stdc++fs            # 这是为了支持std::experimental::filesystem
    # 其他必要的库
)

