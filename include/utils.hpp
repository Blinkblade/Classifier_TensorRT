#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>
#include <system_error>
#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 检查最后一个cuda 内核
static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 判断路径是否存在
bool fileExists(const std::string fileName);
// 通过onnxpath获取/生成enginepath
std::string getEnginePath(std::string onnxPath);
// 取得文件类型/名称
std::string getFileType(std::string filePath);
std::string getFileName(std::string filePath);
std::string printDims(const nvinfer1::Dims dims);
std::string printTensor(float* tensor, int size);
std::string printTensorShape(nvinfer1::ITensor* tensor);
std::string getPrecision(nvinfer1::DataType type);
// 加载engine文件
std::vector<unsigned char> loadFile(const std::string &path);
// 从input image生成output path
std::string changePath(std::string srcPath, std::string relativePath, std::string postfix, std::string tag);




#endif