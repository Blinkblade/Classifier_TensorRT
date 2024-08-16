#include "trt_classifier.hpp"
#include "NvInfer.h"
#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "utils.hpp"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include "imagenet_labels.hpp"
#include "trt_timer.hpp"
namespace model{

namespace classifier{


// 这里我们需要根据传入的plan数据生成上下文,并设置input/output bindings, 分配host/device的memory等
// 创建上下文的流程为: 创建runtime -> runtime 反序列化 生成 engine -> engine 生成上下文 context
void model::classifier::Classifier::setup(void const *data, std::size_t size)
{
    // 把对象的相关成员变量一起设置一下

    // 创建runtime
    this->m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*this->m_logger), destroy_trt_ptr<nvinfer1::IRuntime>);
    // 使用runtime反序列化生成引擎, 从data中读出size的序列
    this->m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(this->m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<nvinfer1::ICudaEngine>);
    // 使用engine创建上下文, 这里包含了所有推理所需的内容
    this->m_context = std::shared_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext(), destroy_trt_ptr<nvinfer1::IExecutionContext>);
    LOGD("===============createExecutionContext SUCCESSFULLY===============");
    // 设置bindings, 分配memory

    // 首先拿到input 和 output形状
    // binding的结构[[list of inputs],[list of outputs]]
    this->m_inputDims = this->m_context->getBindingDimensions(0);
    this->m_outputDims = this->m_context->getBindingDimensions(1);
    // 考虑到大多数classification model都是1 input, 1 output, 这边这么写。如果像BEVFusion这种有多输出的需要修改

    // 计算输入输出的大小
    // hwc*float
    this->m_inputSize = this->mParams->img_info.h * this->mParams->img_info.w * this->mParams->img_info.c * sizeof(float);
    // num_class * float
    this->m_outputSize = this->mParams->numclass * sizeof(float);
    // 计算面积用于一维寻址
    this->m_imgArea = this->mParams->img_info.h * this->mParams->img_info.w;

    // 初始化stream
    CUDA_CHECK(cudaStreamCreate(&this->m_stream));

    // 分配空间
    // 分配host给0
    // 因为输入时**所以指针需要再引用一次
    CUDA_CHECK(cudaMallocHost(&this->m_inputMemory[0], this->m_inputSize));
    CUDA_CHECK(cudaMallocHost(&this->m_outputMemory[0], this->m_outputSize));
    // 分配cuda给1
    CUDA_CHECK(cudaMalloc(&this->m_inputMemory[1], this->m_inputSize));
    CUDA_CHECK(cudaMalloc(&this->m_outputMemory[1], this->m_outputSize));

    //创建m_bindings，之后再寻址就直接从这里找
    // bindings会作为trt推理的输入因此这里使用gpu上的空间
    this->m_bindings[0] = this->m_inputMemory[1];
    this->m_bindings[1] = this->m_outputMemory[1];
    LOGD("===============SETUP CLASSIFIER SUCCESSFULLY===============");

}

// 实现cpu的前处理
bool Classifier::preprocess_cpu()
{
    /*Preprocess -- 获取mean, std*/
    // 使用imagenet的mean std
    // 注意,由于cpu版本的计算在bgr2rgb之后,因此变一下顺序
    // float mean[]       = {0.406, 0.456, 0.485};
    // float std[]        = {0.225, 0.224, 0.229};
    float mean[]       = {0.485, 0.456, 0.406};
    float std[]        = {0.229, 0.224, 0.225};
    // 读入图片
    LOG("Classifier Inference Load Image from : %s", this->m_imagePath.c_str());
    cv::Mat src = cv::imread(this->m_imagePath);
    // 如果读取失败则返回错误
    if (src.data == nullptr) {
        LOGE("ERROR: Image file not founded in bool Classifier::preprocess_cpu()! Program terminated"); 
        return false;
    }
    // 取得目标size
    int tarH = this->mParams->img_info.h;
    int tarW = this->mParams->img_info.w;

    // 记录一下时间
    this->m_timer->start_cpu();

    // 调用已经写好的preprocess, 把结果存放到m_inputMemory[0]里,然后拷贝到gpu上
    preprocess::preprocess_cpu(src,this->m_inputMemory[0], tarH, tarW, mean, std, preprocess::tactics::CPU_BILINEAR);
    // cuda mem cpy,从cpu移动数据到gpu
    CUDA_CHECK(cudaMemcpyAsync(this->m_inputMemory[1], this->m_inputMemory[0], this->m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, this->m_stream));

    this->m_timer->stop_cpu();
    this->m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");

    return true;
}
// classifier里主要设置一下mean std和load image,其他的交给preprocess中的对应函数
bool Classifier::preprocess_gpu()
{
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    // 与this->loadimage区别开,那个只是用来初始化mimagepath的
    LOG("Classifier Inference Load Image from : %s", this->m_imagePath.c_str());
    cv::Mat src = cv::imread(this->m_imagePath);
    if (src.data == nullptr) {
        LOGE("ERROR: Image file not founded in bool Classifier::preprocess_gpu()! Program terminated"); 
        return false;
    }
    // 取得目标size
    int tarH = this->mParams->img_info.h;
    int tarW = this->mParams->img_info.w;

    /*Preprocess -- 测速*/
    m_timer->start_gpu();

    // 调用前处理函数, 处理结果直接写入m_inputMemory[1]
    preprocess::preprocess_resize_cvt_norm_trans_gpu(src, this->m_inputMemory[1], tarH, tarW, mean, std, preprocess::tactics::GPU_BILINEAR);

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}
bool Classifier::postprocess_cpu()
{
    // 此时已经完成推理,在m_outputMemory[1]中已经存放了推理结果
    /*Postprocess -- 测速*/
    this->m_timer->start_cpu();

    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size    = this->mParams->numclass * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(this->m_outputMemory[0], this->m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, this->m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    /*Postprocess -- 寻找label*/
    ImageNetLabels labels;
    int pos = max_element(this->m_outputMemory[0], this->m_outputMemory[0] + this->mParams->numclass) - this->m_outputMemory[0];

    this->m_timer->stop_cpu();
    this->m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    LOG("*** RESULT: Inference result: %s ***", labels.imagenet_labelstring(pos).c_str());     
    return true;
}
bool Classifier::postprocess_gpu() {
    /*
        由于classification task的postprocess比较简单，所以CPU/GPU的处理这里用一样的
        对于像yolo这种detection model, postprocess会包含decode, nms这些处理。可以选择在CPU还是在GPU上跑
    */
    return postprocess_cpu();

}
std::shared_ptr<Classifier> make_classifier(std::string onnx_path, logger::Level level, Params params)
{

    auto classifier = std::make_shared<Classifier>(onnx_path, level, params);
    classifier->init_model();
    return classifier;
}

void Classifier::reset_task(){

};

}
}
