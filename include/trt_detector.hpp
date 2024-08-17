#ifndef __TRT_DETECTOR_HPP__
#define __TRT_DETECTOR_HPP__
// 防止重复引入头文件

#include "trt_model.hpp"

namespace model{
namespace detector{


//不同的检测模型
enum model_type{
    YOLOV5,
    YOLOV8,
};

//检测模型返回bbox因此定义一个结构体存储bbox
struct bbox{
    // xxyy
    float x0,y0,x1,y1;
    // box的置信度
    float confidence;
    // 分类ID
    int label;
    // 判断是否被nms过滤
    bool  flg_remove = false;


    //初始化
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1,float confidence, int label):
        x0(x0),y0(y0),x1(x1),y1(y1),
        confidence(confidence),
        label(label){};
};


//构建一个检测器
class Detector : public model::Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    // 如果子类构造函数什么也不写,就自动调用父类的构造函数
    // 这里的赋值方法就是写个MODEL构造函数然后把值传进去
    Detector(std::string  onnxPath, logger::Level level, Params params):
        Model(onnxPath, level, params){};

// 实现model中的虚函数
public:
    // setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文。
    // 由于不同task的input/output的tensor不一样，所以这里的setup需要在子类实现
    virtual void  setup(void const* data, std::size_t size=0) override;

    virtual void reset_task() override;

    // 不同的task的前处理/后处理是不一样的，所以具体的实现放在子类
    // virtual = 0表示纯虚函数, 基类无定义, 子类必须自己实现
    virtual bool preprocess_cpu() override;
    virtual bool preprocess_gpu() override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override; 

// 设置detector的专有类
private:
    // bbox列表存储检测结果
    std::vector<bbox> m_bboxes;
    int m_inputSize; 
    // 用于计算索引
    int m_imgArea;
    int m_outputSize;

};


// 外部调用接口
std::shared_ptr<Detector> make_detector(std::string  onnxPath, logger::Level level, Params params);

}   //  namespace detector
}   //  namespace model


#endif //__TRT_DETECTOR_HPP__
