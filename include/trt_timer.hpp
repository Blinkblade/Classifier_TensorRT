#ifndef __TRT_TIMER__
#define __TRT_TIMER__
// __TRT_TIMER__ 是预先定义的宏,名字可以随意只要保证可识别性和不重复即可

// 计时库
#include<chrono>
// 比例库
#include<ratio>
#include<iostream>
#include<string>
#include"trt_logger.hpp"
// cuda基本库
#include"cuda_runtime.h"



// 设置计时功能的namespace
namespace timer{


// 设计timer类用于计时
class Timer{
    // 公有属性,是一些比例方便单位变换
    public:
    // using相当于typedef, 给一个类型取别名, 这里s以后就等于std::ratio<1, 1>
    // std::ratio<1, 1000> : 第一个值为分子第二个值为分母, 代表1/1000
    // std::ratio 通常与 C++ 的 std::chrono 库结合使用，用于定义时间持续时间或处理与时间相关的转换。
    // 它不是一个可直接用作数值的类型，而是一种类型构造器，用于定义编译时常数比例。但不能直接作为数值使用
        using s = std::ratio<1,1>;
        using ms = std::ratio<1,1000>;
        using us = std::ratio<1,1000000>;
        using ns = std::ratio<1,1000000000>;
    // 构造函数
    public:
        Timer();
        ~Timer();
    // 公有方法
    public:
    // 开始/结束计时
        void start_cpu();
        void start_gpu();
        void stop_cpu();
        void stop_gpu();

        // 模板函数表示接下来的函数可以传入多种值
        // template表示这是模板, typename表示对数据类型使用模板,
        // span是我们自定义的模板名字,相当于传统模板T,这里span代上面定义的时间尺度
        // 再调用duration_cpu时,需要显示指定span,例如 duration_cpu<ms>("Processing time")
        template <typename span>
        void duration_cpu(std::string msg);
        // gpu计算时间有自己的方法不用模板span了
        void duration_gpu(std::string msg);
    
    // 私有属性
    // _下划线的命名方式表示他们是私有内部变量不希望被外部访问
    private:
    // high_resolution_clock类型的时间点
        std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
        std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
    // cudaenvent记录cuda事件的时间
        cudaEvent_t _gStart;
        cudaEvent_t _gStop;
        // _timeElasped通常被用于记录gpu时间
        float _timeElasped;
};

// 使用模板类通常再hpp中就定义好
template <typename span>
inline void Timer::duration_cpu(std::string msg)
{
    // 查看当前使用的是什么单位
    std::string str;
    if(std::is_same<span, s>::value) { str = "s"; }
    else if(std::is_same<span, ms>::value) { str = "ms"; }
    else if(std::is_same<span, us>::value) { str = "us"; }
    else if(std::is_same<span, ns>::value) { str = "ns"; }


    // 自带duration计算用时,输入span指定单位
    std::chrono::duration<double, span> time = _cStop - _cStart;
    // %-60S六十个字符宽,.6lf六位double类型浮点数
    LOGV("%-60s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}

}//namespace timer

#endif