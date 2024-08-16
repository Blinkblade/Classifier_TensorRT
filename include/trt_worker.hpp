#ifndef __TRT_WORKER_HPP__
#define __TRT_WORKER_HPP__

#include <string>
#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"

// 创建一个worker用于调用所有的模型执行任务,
// 虽然现在只有一个分类器
namespace worker{

// 这是一个工厂类,封装所有需要用到的功能
// 各个功能单独实现,最后由worker调用
class Worker{
public:

    // worker的构造函数应该包含全部任务的初始化参数
    // 由于现在只有分类器因此和classifer的初始化参数一样
    Worker(std::string onnxPath, logger::Level level, model::Params params);
    // 一键推理
    void inference(std::string imagePath);


protected:
    // worker中可以包含任务所需的所有模型
    // 用指针存储这些对象, 实现动态绑定/减少耦合/易于释放资源/管理生命周期
    std::shared_ptr<logger::Logger>          m_logger;
    std::shared_ptr<model::Params>           m_params;
    // 因为今后考虑扩充为multi-task，所以各个task都是worker的成员变量
    std::shared_ptr<model::classifier::Classifier>  m_classifier;

    std::shared_ptr<model::detector::Detector> m_detector;

};

// 提供外部构建接口
std::shared_ptr<Worker> create_worker(std::string onnxPath, logger::Level level, model::Params params);



} // namespace worker

#endif // __TRT_WORKER_HPP__