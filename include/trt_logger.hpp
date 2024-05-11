#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include<iostream>
#include "NvInfer.h"
#include <string>
// #include <stdarg.h>
#include <memory>

// 可变宏,...代表变长输入
#define LOGF(...) logger::Logger::__log_info(logger::Level::FATAL, __VA_ARGS__)
#define LOGE(...) logger::Logger::__log_info(logger::Level::ERROR, __VA_ARGS__)
#define LOGW(...) logger::Logger::__log_info(logger::Level::WARN,  __VA_ARGS__)
#define LOG(...)  logger::Logger::__log_info(logger::Level::INFO,  __VA_ARGS__)
#define LOGV(...) logger::Logger::__log_info(logger::Level::VERB,  __VA_ARGS__)
#define LOGD(...) logger::Logger::__log_info(logger::Level::DEBUG, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

namespace logger{

// 枚举一个log等级,越小越严重,越大越
enum class Level : int32_t{
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    VERB  = 4,
    DEBUG = 5
};
// 继承nvinfer的Ilogger,
// Ilogger没有内容,需要我们自己定义
class Logger : public nvinfer1::ILogger{
    
public:
    // 构造函数
    Logger();
    // 输入log等级初始化logger
    Logger(Level Level);
    // 重写虚函数时最好加上override
    virtual void log(Severity severity, const char* msg) noexcept override;
    // 在这里实现具体的log功能供log调用
    // 这是一个可变参数的函数
    // 这个函数的名字和参数表明它用于日志记录：

    // Level level：一个枚举或整型，指定日志信息的级别（例如，错误、警告、信息等）。
    // const char format*：一个格式字符串，与 printf 类似，指定后续参数如何格式化输出。
    // ...（省略号）：表示函数接受可变数量的参数，这些参数将根据 format 字符串进行格式化。
    // 静态方法
    static void __log_info(Level level, const char* format, ...);

    // 返回成员变量值
    Level get_level();
    Severity get_severity();
    // 交叉返回
    Level get_level(Severity severity);
    Severity get_severity(Level level);

// 成员变量
private:
// 对应成员变量最好有函数可以拿到值
    static Level m_level;
    Severity m_severity;

};

// 提供外部调用接口
std::shared_ptr<Logger> create_logger(Level level);

} // namespace logger

#endif //__LOGGER_HPP__