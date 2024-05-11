#include "trt_timer.hpp"
#include<chrono>

#include "utils.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "trt_logger.hpp"

namespace timer{

// 构造函数实现 
// 设置当前时间用于初始化
timer::Timer::Timer()
{
    this->_timeElasped = 0;
    this->_cStart = std::chrono::high_resolution_clock::now();
    this->_cStop = std::chrono::high_resolution_clock::now();
    // gpu时间
    cudaEventCreate(&this->_gStart);
    cudaEventCreate(&this->_gStop);

}

// 注意,对象销毁时需要释放cuda资源
// 销毁顺序和生成顺序最好对称
Timer::~Timer()
{
    // 释放事件指针
    cudaFree(this->_gStop);
    cudaFree(this->_gStart);

    // 销毁事件对象
    cudaEventDestroy(this->_gStop);
    cudaEventDestroy(this->_gStart);

}

void Timer::start_cpu()
{
    // 获取开始时间
    this->_cStart = std::chrono::high_resolution_clock::now();
}

void Timer::start_gpu()
{
    // 使用cuda事件记录
    // 输入参数时cudaevent和stream, 0表示默认cuda流
    cudaEventRecord(this->_gStart,0);
}
void Timer::stop_cpu()
{
    // 获取结束时间
    this->_cStop = std::chrono::high_resolution_clock::now();
}
void Timer::stop_gpu()
{
    // 使用cuda事件记录
    // 输入参数时cudaevent和stream
    cudaEventRecord(this->_gStop,0);
}
void Timer::duration_gpu(std::string msg)
{
    // 事件同步保证数据正确
    CUDA_CHECK(cudaEventSynchronize(this->_gStart));
    CUDA_CHECK(cudaEventSynchronize(this->_gStop));
    // 使用cuda中自带的计算时间差方法
    // &传入因为要改变值
    cudaEventElapsedTime(&this->_timeElasped, this->_gStart, this->_gStop);
    LOGV("%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
}

}
