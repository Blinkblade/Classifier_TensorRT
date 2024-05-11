#ifndef __TEST_CUDA__
#define __TEST_CUDA__

#include "cuda_runtime_api.h"

namespace cutest {
    // 定义一个cuda语法的创建函数,
    // 在cu文件中引入该头文件实现细节
    void timer_gpu_create();

    // 定义一个不含cuda语法的调用函数
    // 在cpp文件中实现细节
    void timer_gpu_test();
}//namespace cutest 

#endif //__TEST_CUDA__
