#ifndef __TRT_CLASSIFIER_HPP__
#define __TRT_CLASSIFIER_HPP__

#include "trt_model.hpp"


// 继承父类时,namespace设置成 父类::子类 更有层次感
namespace model{
namespace classifier{

//  : public --> 公有继承
class Classifier : public Model{
public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    // 如果子类构造函数什么也不写,就自动调用父类的构造函数
    // 这里的赋值方法就是写个MODEL构造函数然后把值传进去
    Classifier(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

// 需要重写的函数全部单独实现一遍
public:
    // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
    // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
    virtual void  setup(void const* data, std::size_t size=0) override;

    // 不同的task的前处理/后处理是不一样的，所以具体的实现放在子类
    // virtual = 0表示纯虚函数, 基类无定义, 子类必须自己实现
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

    virtual void reset_task() override;


// 自有成员变量:
private:
    // 计算输入输出大小,用于分配内存
    int m_inputSize;     
    int m_outputSize;
    // 计算图像面积,用于组织一维寻址
    int m_imgArea;   
};


// 外部调用的接口
std::shared_ptr<Classifier> make_classifier(
    std::string onnx_path, logger::Level level, Params params);

} // namespace classifier
} // namespace model









#endif //__TRT_CLASSIFIER__