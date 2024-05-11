#include "trt_worker.hpp"
#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_classifier.hpp"
namespace worker{

worker::Worker::Worker(std::string onnxPath, logger::Level level, model::Params params)
{
    // 初始化logger
    this->m_logger = logger::create_logger(level);

    // 根据传入的参数组的任务类型创建该参数组的模型
    if(params.task == model::task_type::CLASSIFICATION){
        // 分类任务就构建分类器
        this->m_classifier = model::classifier::make_classifier(onnxPath, level, params);
    }
}

// 实现一键推理,将i所需参数传入对应模型即可
void Worker::inference(std::string imagePath)
{
    // 只要对应的任务模型存在就设置参数并推理
    if(this->m_classifier != nullptr){
        this->m_classifier->load_image(imagePath);
        this->m_classifier->inference();
    }
}

// 提供外部调用接口 
std::shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params) 
{
    // 使用智能指针来创建一个实例
    return std::make_shared<Worker>(onnxPath, level, params);
}

}// namespace worker
