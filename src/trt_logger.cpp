#include "trt_logger.hpp"
#include "NvInfer.h"
// va_start
#include <stdarg.h>
#include<iostream>

namespace logger{

// 为类初始化一个mlevel,如果实例化对象则会被替换成初始化的level
// 如果直接使用类方法则是info
Level Logger::m_level = Level::INFO;

// 构造函数
Logger::Logger(Level Level)
{
    this->m_level = Level;
}

// 处理信息的函数
void Logger::log(Severity severity, const char *msg) noexcept {
    /*  
        设置日志输出等级
        - 对FATAL, ERROR, WARNING级别的日志正常打印 
        - TensorRT的log级别如果是INFO或者是VERBOSE的时候，只有当logger的level在大于VERBOSE的时候再打出
    */
   if((severity <= this->get_severity(Level::WARN)) || this->m_level >= Level::DEBUG){
    // “%s”msg的解析类型,输入的msg会以字符串形式被解析
        this->__log_info(get_level(severity), "%s", msg);
   }
}

// Level level：一个枚举或整型，指定日志信息的级别（例如，错误、警告、信息等）。
// const char format*：一个格式字符串，与 printf 类似，指定后续参数如何格式化输出。
// ...（省略号）：表示函数接受可变数量的参数，这些参数将根据 format 字符串进行格式化。
void Logger::__log_info(Level level, const char* format, ...) {
    
    // 日志消息缓冲区，最大长度为 1000
    char msg[1000];
    // 可变参数列表变量
    va_list args;
    // 初始化可变参数列表，读取 format 后传入的参数列表
    va_start(args, format);
    // 初始化变量 n，用于记录当前字符串的长度
    int n = 0;
    
    // 根据日志级别添加不同颜色的前缀
    switch (level) {
        case Level::DEBUG:
            // 为 DEBUG 级别添加绿色的 [debug] 前缀
            // snprintf 会根据 format 字符串将数据格式化到 str 指向的缓冲区，但它最多只会写入 size-1 个字符，
            // 以确保总是有空间放置字符串终止符 '\0'。函数返回值是如果不受 size 限制会写入的字符总数（不包括终止符）。
            // str：指向用于存储输出字符串的缓冲区的指针。
            // size：缓冲区的最大大小（包括最后的空字符 '\0'）。
            // format：一个格式字符串，指定后续参数如何格式化。
            // ...：一个可变参数列表，提供额外的数据来填充格式字符串。
            // msg+n就是msg[n]
            n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR);
            break;
        case Level::VERB:
            // 为 VERB 级别添加紫色的 [verb] 前缀
            n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR);
            break;
        case Level::INFO:
            // 为 INFO 级别添加黄色的 [info] 前缀
            n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR);
            break;
        case Level::WARN:
            // 为 WARN 级别添加蓝色的 [warn] 前缀
            n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR);
            break;
        case Level::ERROR:
            // 为 ERROR 级别添加红色的 [error] 前缀
            n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR);
            break;
        default:
            // 对于未定义的级别，默认添加红色的 [fatal] 前缀
            n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR);
            break;
    }
    
    // 使用可变参数列表和格式字符串，将格式化的消息附加到 msg 缓冲区
    // vsnprintf是可变的snprintf
    // 这两步就是先写 等级标志，然后再写入对应的日志
    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);
    
    // 结束可变参数处理
    va_end(args);

    // if (m_level == logger::Level::INFO){
    //     std::cout<< "=======INFO========" <<std::endl;
    // }    

    // 判断当前日志级别是否低于或等于全局日志级别
    if (level <= m_level) {
        // 如果符合条件，则输出日志消息到标准输出
        fprintf(stdout, "%s\n", msg);
    }
    
    // 如果日志级别为 ERROR 或更低级别，则刷新输出缓冲区并终止程序
    if (level <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}
// 空输入得到当前值
Level Logger::get_level()
{
    return this->m_level;
}

Logger::Severity Logger::get_severity()
{
    return this->m_severity;
}

// 交叉输入可以获取对应值
Level Logger::get_level(Severity severity)
{
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return Level::FATAL;
        case Severity::kERROR:          return Level::ERROR;
        case Severity::kWARNING:        return Level::WARN;
        case Severity::kINFO:           return Level::INFO;
        case Severity::kVERBOSE:        return Level::VERB;
    }
}

// 继承父类的空间后可以直接用自己作为空间
Logger::Severity Logger::get_severity(Level level)
{
    switch (level) {
        case Level::FATAL: return Severity::kINTERNAL_ERROR;
        case Level::ERROR: return Severity::kERROR;
        case Level::WARN:  return Severity::kWARNING;
        case Level::INFO:  return Severity::kINFO;
        case Level::VERB:  return Severity::kVERBOSE;
        default:           return Severity::kVERBOSE;
    }
}

// 交叉输入返回对应值

std::shared_ptr<Logger> create_logger(Level level) {
    return std::make_shared<Logger>(level);
};



}
