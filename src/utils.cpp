#include "utils.hpp"
#include <string>
#include "trt_logger.hpp"
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <sstream>
#include "NvInfer.h"
#include "trt_model.hpp"
#include <vector>


using std::cout;
using std::endl;

// 判断路径是否存在
bool fileExists(const std::string fileName) {
    if (!std::experimental::filesystem::exists(
            std::experimental::filesystem::path(fileName))){
        return false;
    }else{
        return true;
    }
};

// find和substr组合获取文件名
// dir/dir2/name.type
std::string getFileType(std::string filePath){
    int pos = filePath.rfind(".");
    std::string suffix;
    suffix = filePath.substr(pos, filePath.length());
    return suffix;
}

std::string getFileName(std::string filePath){
    int pos = filePath.rfind("/");
    std::string suffix;
    suffix = filePath.substr(pos + 1, filePath.length());
    return suffix;
}

std::string printDims(const nvinfer1::Dims dims)
{
    int n = 0;
    char buff[100];
    std::string result;

    // 向buff中输入, sizeof(buff) - n指还能写多少(maxsize), "[ "为写入值
    // snprintf返回写入的末尾方便继续写入
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    // 逐个写入
    for (int i = 0; i < dims.nbDims; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1) {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

std::string printTensor(float *tensor, int size)
{
    int n = 0;
    char buff[100];
    std::string result;
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < size; i++){
        n += snprintf(buff + n, sizeof(buff) - n, "%8.4lf", tensor[i]);
        if (i != size - 1){
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

std::string printTensorShape(nvinfer1::ITensor *tensor)
{
    std::string str;
    str += "[";
    auto dims = tensor->getDimensions();
    for (int j = 0; j < dims.nbDims; j++) {
        str += std::to_string(dims.d[j]);
        if (j != dims.nbDims - 1) {
            str += " x ";
        }
    }
    str += "]";
    return str;
}

// 枚举类型本质上是index,无法直接打印,转换成string才好
std::string getPrecision(nvinfer1::DataType type)
{
    switch(type) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        case nvinfer1::DataType::kUINT8:  return "UINT8";
        default:                          return "unknown";
    }
}

// 从engine path中加载engine文件
std::vector<unsigned char> loadFile(const std::string &path)
{
    // 读文件可以用fstream
    // 输入为 文件路径 写入位置(std的ios的输入流) 二进制
    // 指定 ios::in | ios::binary 标志，表示以输入（读取）模式打开文件，
    // 并且以二进制模式处理文件内容，避免任何数据的转换（如换行符转换）。
    std::ifstream in(path, std::ios::in | std::ios::binary);
    // 如果输入流没有打开,说明读入失败返回空
    if (!in.is_open()){
        return {};
    }
    // in.seekg(0, ios::end) 将文件指针移动到文件末尾。
    // in.tellg() 返回当前文件指针的位置，这里是文件末尾，因此返回值表示文件的总长度（字节数）。
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();


    // 声明一个 vector<uint8_t> 类型的变量 data，用于存储文件数据。
    // 注意 uint8_t 是 unsigned char 的同义词，所以这里和函数返回类型是一致的。
    // 如果文件长度大于0，先将文件指针重新定位到文件开头 (ios::beg)。
    // 通过 data.resize(length) 调整 data 的大小，确保有足够的空间存储文件内容。
    // 使用 in.read((char*)&data[0], length) 从文件中读取 length 字节的数据到 data 的内存地址开始的位置。
    // 这里需要将 data[0] 的地址转为 char* 类型，因为 read 函数需要一个 char* 类型的指针。
    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }

    // 关闭流
    in.close();
    return data;
}

std::string getEnginePath(std::string onnxPath)
{
    // std的string自带substr可以直接切片
    // 参数为下标,因此用自带的find rfind自动生成对应下标

    // 路径结构为: dirs/onnx/name.onnx ===> dirs/engine/name.engine
    // 先取得模型名字的位置,rfind是从右网左找第一个
    int name_l = onnxPath.rfind("/");
    int name_r = onnxPath.rfind(".");
    // 很显然在/和.之间的substr就是name
    // l r代表left right, substr第二个参数是长度
    std::string name = onnxPath.substr(name_l , name_r - name_l);
    // models/onnx/resnet34.onnx
    // cout << "=========从 " << name_l << " 到 " << name_r << " =========" <<endl;
    LOGD("=========get engine path name : %s ===========", name.c_str());
    // 取得文件夹
    int dir_r = onnxPath.find("/");
    int dir_l = 0;
    std::string dir = onnxPath.substr(dir_l, dir_r);


    // 组合成engine name:
    std::string enginePath = dir + "/engine" + name + ".engine";

    // 打印一下看看正不正确
    cout << "=========从 " + onnxPath + " 生成 " + enginePath + " =========" <<endl;

    return enginePath;
};