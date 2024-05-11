#ifndef __TRT_MODEL_HPP__
#define __TRT_MODEL_HPP__
// 防止重复引入头文件

#include <iostream>
// 用于分配/读写内存
#include <memory>
// 基本数据结构
#include <vector>
#include <string>
// nvidia推理库
#include "NvInfer.h"
#include "trt_timer.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

// 设置工作空间大小为256MB
#define WORKSPACESIZE 1<<28

namespace model{
enum task_type {
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
};

enum device {
    CPU,
    GPU
};

enum precision {
    FP32,
    FP16,
    // INT8， TODO
};

// 每个model都要有自己的mParams
struct image_info
{
    int h;
    int w;
    int c;
    // 结构体初始化,用int height, int width, int channel初始化hwc
    image_info(int height, int width, int channel) : 
    h(height), w(width), c(channel){};

};

/* 构建一个针对trt的shared pointer. 所有的trt指针的释放都是通过ptr->destroy完成*/
template <typename T>
void destroy_trt_ptr(T* ptr){

    if(ptr){
        // 销毁时提示一下
        std::string type_name = typeid(T).name();
        LOGD("Destroy %s", type_name.c_str());
        ptr->destroy(); 
    }

}


// 每个model含有的内部参数组
// 设置一个默认的初始化值
// struct mParam
// {
//     // 默认使用gpu
//     device dev              = GPU;
//     // imagenet分类数
//     int numclass            = 1000;
//     // 任务类型
//     task_type task          = CLASSIFICATION;
//     // 工作空间大小
//     int ws_size             = WORKSPACESIZE;
//     // 默认精度
//     precision prec          = FP32;
//     // 默认图像形状,传统分类器输入
//     image_info img_info     = {224, 224, 3};
//     // 插值方式
//     preprocess::tactics tac = preprocess::tactics::GPU_BILINEAR;
//     /* data */
// };

struct Params
{
    // 默认使用cpu
    device dev              = CPU;
    // imagenet分类数
    int numclass            = 1000;
    // 任务类型
    task_type task          = CLASSIFICATION;
    // 工作空间大小
    int ws_size             = WORKSPACESIZE;
    // 默认精度
    precision prec          = FP32;
    // 默认图像形状,传统分类器输入
    image_info img_info     = {224, 224, 3};
    // 插值方式
    preprocess::tactics tac = preprocess::tactics::CPU_BILINEAR;

};



//  由于model是纯虚函数(存在virtual = 0,因此不能直接创建对象,需要继承)
class Model{
    public: 
    // 构造析构,析构不实现有需要的再自行实现
    // // 构造时初始化onnxpath logger timer param workspace enginepath
        Model(std::string  onnxPath, logger::Level level, Params params);
        virtual ~Model(){};
        // 加载图像
        void load_image(std::string image_path);
        // 初始化模型,包括从onnx build和从engine load
        void init_model();
    
    // 具体的功能函数
    public:
        // 从onnx构建模型
        bool build_engine();
        // 从已有engine load模型
        // 都是直接读取成员变量
        bool load_engine();
        // 使用engine推理,包括前处理和后处理和enqueue_bindings
        void inference();
        // dnn 推理部分,不包括前处理和后处理
        bool enqueue_bindings();
        // 保存序列化的engine
        void save_plan(nvinfer1::IHostMemory& plan);
        // 打印静态网络结构
        void print_network(nvinfer1::INetworkDefinition &network, bool optimized);

    // 需要子类自己实现的虚函数
    public:
        // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
        // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
        virtual void  setup(void const* data, std::size_t size=0);

        // 不同的task的前处理/后处理是不一样的，所以具体的实现放在子类
        // virtual = 0表示纯虚函数, 基类无定义, 子类必须自己实现
        virtual bool preprocess_cpu()      = 0;
        virtual bool preprocess_gpu()      = 0;
        virtual bool postprocess_cpu()     = 0;
        virtual bool postprocess_gpu()     = 0; 

    

    // 成员变量,想想实现模型构建/加载/推理都需要哪些参数
    // m_表示这是一个成员变量,但是可以修改,_一般表示不愿被修改的参数
    // private:之所以用protected是因为我们希望model的派生类可以直接访问这些变量
    // 其实用的还是挺多的, private不太方便
    protected:
        // onnx的路径
        std::string m_onnxPath;
        // engine路径
        std::string m_enginePath;
        // 图像路径
        std::string m_imagePath;


        // 用于trt引擎的配置
        // 工作空间大小
        int m_workspaceSize;

        // 输入的binds,binds = [inputs ,outputs]
        // float* [2]代表 [[floats],[floats]],这里分别指代inputs和outputs
        float* m_bindings[2];
        // 这里的 float* [2], 0代表cpu的存储, 1代表cuda存储
        float* m_inputMemory[2];
        float* m_outputMemory[2];

        // 结构体最好用引用方式,防止循环嵌套
        // 此外,指针类型还代表该成员并不被对象所完全拥有,对象只是使用,二者生命周期并不绝对关联
        Params* mParams;

        // 输入输出维度
        // nvinfer1::Dims
        // 这个类用来表示一个张量的维度。它包含两个主要的成员：
        // int nbDims：表示维度的数量。例如，对于一个 3D 张量，nbDims 将是 3。
        // int d[MAX_DIMS]：一个整数数组，存储每个维度的大小。MAX_DIMS 是 TensorRT 定义的一个常量，通常足以覆盖大多数应用场景。
        // nvinfer1::Dims 是一个灵活的类，可以用来描述任何形状的张量，从而使得 TensorRT 的接口能够处理各种不同的数据布局。
        nvinfer1::Dims m_inputDims;
        nvinfer1::Dims m_outputDims;
        // 线程序号
        cudaStream_t m_stream;

        // 对于类型为 其他类 的成员变量, 最好使用指针创建, 可以有效防止多重嵌套
        // 这里有些变量本身就是一个指针, 使用shared智能指针可以维持一个不唯一引用的成员指针

        // logger用于创建Builder等trt成员
        std::shared_ptr<logger::Logger>                 m_logger;
        // timer用于计时评估
        std::shared_ptr<timer::Timer>                   m_timer;
        // engine用于序列化模型与推理
        std::shared_ptr<nvinfer1::ICudaEngine>          m_engine;
        // runtime用于反序列化
        std::shared_ptr<nvinfer1::IRuntime>             m_runtime;
        // 推理上下文,保存了推理时的各种信息
        std::shared_ptr<nvinfer1::IExecutionContext>    m_context;
        // network,用于静态存储从onnx读取的网络结构
        std::shared_ptr<nvinfer1::INetworkDefinition>   m_network;


};


} //namespace model








#endif //__TRT_MODEL_HPP__