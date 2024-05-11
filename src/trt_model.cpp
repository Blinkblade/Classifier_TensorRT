#include "trt_model.hpp"
#include "trt_logger.hpp"
#include "utils.hpp"
#include "trt_timer.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"

namespace model{

// 构造函数
// 构造时初始化onnxpath logger timer param workspace enginepath
Model::Model(std::string onnxPath, logger::Level level, Params params)
{   
    this->m_onnxPath = onnxPath;
    // cpp11中已经支持使用makeshared初始化一个shared智能指针
    // 这种方法的初始化相当于使用T的构造函数,按照其构造函数方法传参即可
    this->m_logger = std::make_shared<logger::Logger>(level);
    // timer空初始化
    this->m_timer = std::make_shared<timer::Timer>();
    // 初始化m_params
    this->mParams = new model::Params(params);
    // worksize
    this->m_workspaceSize = WORKSPACESIZE;
    // onnx path和eninepath相关联
    this->m_enginePath = getEnginePath(this->m_onnxPath);


}

void Model::load_image(std::string image_path)
{
    // 如果路径存在
    if (fileExists(image_path)){
        this->m_imagePath = image_path;
        // 更习惯打印完整路径
        LOG("Model:      %s", m_onnxPath.c_str());
        LOG("Image:      %s", m_imagePath.c_str());
    }
    else{
        // 如果路径不存在
        LOGE("%s not found", image_path.c_str());
    }
}

// 初始化模型,生成推理引擎,包括Build和load两种模式
// 如果上下文已经加载好就不需要再读取直接用上下文即可,因此此时什么也不做
void Model::init_model()
{
    if(this->m_context == nullptr){

        if(fileExists(this->m_enginePath)){
            // 打印一些log用于调试
            LOG("No context but Engine exist, loading engine form %s !",this->m_enginePath.c_str());
            this->load_engine();
            // 追踪状态成功了就说一声
            LOG("Load engine  %s success!",this->m_enginePath.c_str());
        }
        else{
            // 打印一些log用于调试
            LOG("No context and No engine exist, building engine form %s !",this->m_onnxPath.c_str());
            this->build_engine();
            // 追踪状态成功了就说一声
            LOG("Build engine  %s success!",this->m_enginePath.c_str());
        }
    }


}
// build的流程->创建builder->创建network->创建config->创建parser
// parser解析->创建engine->序列化engine为plan->存储序列化plan
// 我们也希望在build一个engine的时候就把一系列初始化全部做完，
// 这部分内容由于不同Model的输入输出不一样,因此再setup中完成,其中包括
//  1. build一个engine
//  2. 创建一个context
//  3. 创建推理所用的stream
//  4. 创建推理所需要的device空间
// 这样，我们就可以在build结束以后，就可以直接推理了。这样的写法会比较干净
bool Model::build_engine()
{   
    // 这些过程很重要容易出错,多写点log追踪
    LOG("===============Start Building Engine in --bool Model::build_engine()--===============");
    
    // 用auto接收智能指针
    // 引用指针需要先访问,即&的操作对象是本体而不是指针
    // 使用destory_trt_ptr这样指针销毁时可以得到提示
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*this->m_logger), model::destroy_trt_ptr<nvinfer1::IBuilder>);
    // 打印一些debug级别的细节log
    LOGD("===============createInferBuilder SUCCESSFULLY===============");
    // 使用builder创建network,1代表静态batch1, network是静态的模型描述,从onnx加载后解析到这里
    // 这里的 1 对应于 1ULL << 0，也就是 nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH。
    // 这意味着你在创建网络时显式指定了批量大小，这通常是处理固定批量大小的推理时所必需的。
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1),model::destroy_trt_ptr<nvinfer1::INetworkDefinition>);
    LOGD("===============createNetworkV2 SUCCESSFULLY===============");
    // 创建config,config用于配制推理的各种配置
    auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(),model::destroy_trt_ptr<nvinfer1::IBuilderConfig>);
    LOGD("===============createBuilderConfig SUCCESSFULLY===============");
    // 创建解析器用于解析onnx模型生成network,IParser是nvonnxparser中的类
    auto parser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *this->m_logger), model::destroy_trt_ptr<nvonnxparser::IParser>);
    LOGD("===============createParser SUCCESSFULLY===============");
    // 至此构建好parser,接下来设置config并利用parser解析onnx写入network,注意,此时network已经和parser绑定
    // 设置工作空间
    config->setMaxWorkspaceSize(this->m_workspaceSize);
    // 开发的时候可以调高一点, 没什么问题了再改成简单日志
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    // 如果解析失败就直接推出并写对应日志,verb设置为1,越大日志越详细,0代表输出所有的debug日志
    if(!parser->parseFromFile(this->m_onnxPath.c_str(), 1)){
        LOGE("===============parseFromFile %s Failed!===============", this->m_onnxPath.c_str());
        return false;
    }

    // 如果设置了FP16精度,就设置相关flag
    // INT8还没做
    if (builder->platformHasFastFp16() && this->mParams->prec == model::FP16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    // 到这里network已经存储了onnx中获取的静态模型,接下来将他转换为engine并序列化存储
    // 最终写入上下文, 这样整个model的推理引擎就存在到了context中

    // 使用builder从network和config中创建引擎
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), model::destroy_trt_ptr<nvinfer1::ICudaEngine>);
    LOGD("===============buildEngineWithConfig SUCCESSFULLY===============");
    // 从config network中序列化网络准备存储engine
    // plan的返回值是nvinfer1::IHostMemory*,即一片host内存的指针
    auto plan = builder->buildSerializedNetwork(*network, *config);
    LOGD("===============buildSerializedNetwork plan SUCCESSFULLY===============");
    // 创建执行器用于反序列化解析出模型并存储到上下文中
    auto runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*this->m_logger));



    save_plan(*plan);
    LOGD("===============save_plan to %s SUCCESSFULLY===============", this->m_enginePath);
    // 根据runtime初始化engine, context, 以及memory
    // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
    // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
    setup(plan->data(), plan->size());

    // 把优化前和优化后的各个层的信息打印出来
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);
    // 到这一步说明build成功,返回True
    LOG("Build engine %s from %s SUCCESSFULLY!",this->m_enginePath,this->m_onnxPath);
    return true;
}

// build load所有类都一样, 因此再父类中给出定义
// load流程: 反序列化engine -> 直接setup
bool Model::load_engine()
{
    // 进入Load说明enginepath被判定为存在
    if (!fileExists(this->m_enginePath)) {
        LOGE("Engine %s does not exits! Program terminated", this->m_enginePath);
        return false;
    }
    // 为什么使用uint8_t来存储序列?
    // uint8_t 是一个无符号的 8 位整数类型，表示可以存储一个字节的数据（从 0 到 255 的值）。和unsigned char是同义
    // 在处理二进制文件时，通常我们关心的是文件的原始字节数据，而不是这些字节所表示的字符或数值。uint8_t 提供了一种简单的方式来直接操作这些字节。
    // 二进制文件，比如一个已编译的 TensorRT 引擎（.engine 文件），是由字节构成的，不应该被解释为具有任何特定字符编码的文本。
    // 使用 uint8_t 类型来处理二进制文件是一种常见且适当的做法，它使得数据操作既清晰又具有类型安全性。这种做法在文件 I/O、网络通信和低级数据处理中非常普遍。
    std::vector<uint8_t> modelData = loadFile(this->m_enginePath);

    // 根据runtime初始化engine, context, 以及memory
    setup(modelData.data(), modelData.size());

    return true;
}

// inference和equeue也很一致,从基类实现
// inference 流程: preprocess-> enqueue -> postprocess
void Model::inference()
{
    if(this->mParams->dev == model::CPU){
        LOG("mParams->dev is CPU, preprocess on CPU!");
        this->preprocess_cpu();
    }
    else{
        LOG("mParams->dev is GPU, preprocess on GPU!");
        this->preprocess_gpu();
    }
    LOG("=================MODEL ENQUEUE START================");
    this->enqueue_bindings();

    if(this->mParams->dev == model::CPU){
        
        this->postprocess_cpu();
        LOG("mParams->dev is CPU, FINISHED preprocess on CPU!");
    }
    else{
        this->postprocess_gpu();
        LOG("mParams->dev is GPU, FINISHED preprocess on GPU!");
    }
}

// 就是吧bindings装到engine里推理,从基类实现
// 推理是使用上下文来做的,成员变量中的context是在setup里根据plan生成的
// 也就是说,只要保存了context就已经可以实现推理了,
// 因为context中保存了推理所需要的所有上下文
bool Model::enqueue_bindings()
{
    // 计算推理时间
    // 推理是在gpu上进行的
    this->m_timer->start_gpu();
    // void** bindings：这是一个指针数组，每个指针指向一个内存缓冲区。这些缓冲区通常包括网络的输入和输出缓冲区。
    // void* 类型用于表示一般性指针，而 void** 表示指向这些通用指针的指针，可以理解为指针的数组。
    // 在这种情况下，它允许传递一个动态大小的数组，数组中的每个元素都是一个指向特定内存区域（输入/输出数据）的指针。
    // cudaStream_t stream：这是一个 CUDA 流，用于管理 GPU 上的异步操作。
    // cudaEvent_t* inputConsumed（可选参数，这里未使用）：这是一个指向 CUDA 事件的指针，用于通知输入缓冲区何时可以安全地重新使用。
    // 如果传递了非空指针，enqueueV2 会在输入数据被 GPU 完全消耗后标记这个事件。
    if(!this->m_context->enqueueV2((void**)this->m_bindings, this->m_stream, nullptr)){
        // 如果推理失败,则写个日志通知大家失败了
        LOGE("Error happens during DNN inference part(enqueue_bindings), program terminated");
        return false;
    }
    // 停止计时并显示计时
    this->m_timer->stop_gpu();
    this->m_timer->duration_gpu("TensorRT DNN inferer on GPU");

    return true;
}

void Model::save_plan(nvinfer1::IHostMemory &plan)
{   
    // 打开engine文件, mode为write binary即写入二进制
    auto f = fopen(this->m_enginePath.c_str(),"wb");
    // 将plan的数据写到f里
    // 1 表示数据以 1 byte 为基本单位组织
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}
// 逐层打印network的layers
// 不清楚可以对照nvidia nvinfer1::INetworkDefinition 的文档写
void Model::print_network(nvinfer1::INetworkDefinition &network, bool optimized)
{
    // 取得所有的输入输出个数
    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    // 所有层数
    // 注意,engine和network的layer不同,共用一个会越界
    int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();

    // 打印所有输入shape:
    for(int i = 0; i < inputCount; i++){
        // 指针类型的都用auto接受
        auto input = network.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }
    // 打印所有outpu shape
    for(int i = 0; i < outputCount; i++){
        auto output = network.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = std::shared_ptr<nvinfer1::IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }


}

void Model::setup(void const *data, std::size_t size)
{
}

}
