#include<string>
#include"trt_timer.hpp"
#include"cuda_runtime_api.h"
#include <cmath> // For sqrt in the kernel
#include<test_cuda.hpp>
// kernel.cu
namespace cutest{

__global__ void dummyKernel(int n, float *x) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        x[index] = sqrt(x[index]);
    }
};

// 测试 GPU 计时是否正常工作
void timer_gpu_create() {
    std::string msg = "GPU computation";
    timer::Timer tmr = timer::Timer();

    // 分配内存
    int n = 1 << 20; // 大约 1M 个元素
    float *d_x;
    cudaMalloc(&d_x, n * sizeof(float));

    // 填充数据
    float *x = new float[n];
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
    }
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 GPU 计时
    tmr.start_gpu();

    // 启动内核，足够的线程来保证可以测量到时间
    dummyKernel<<<(n + 255) / 256, 256>>>(n, d_x);
    cudaDeviceSynchronize(); // 确保内核执行完成

    // 停止 GPU 计时
    tmr.stop_gpu();

    // 输出计时结果
    tmr.duration_gpu(msg);

    // 清理资源
    cudaFree(d_x);
    delete[] x;
}


}// namespace cutest

