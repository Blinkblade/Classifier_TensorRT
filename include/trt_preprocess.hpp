#ifndef __TRT_PREPROCESS__
#define __TRT_PREPROCESS__

#include<iostream>
// 包含OpenCV绝大部分功能和全部核心功能
// 通常这种做法会增加编译时间,但由于opencv不算特别大,因此可以接受
#include "opencv2/opencv.hpp" 
#include "trt_logger.hpp"


namespace preprocess{

// 不同的插值策略
enum class tactics : int32_t {
    CPU_NEAREST         = 0,
    CPU_BILINEAR        = 1,
    GPU_NEAREST         = 2,
    GPU_NEAREST_CENTER  = 3,
    GPU_BILINEAR        = 4,
    GPU_BILINEAR_CENTER = 5,
};

// 输入参数图片 均值mean 标准差std , 一维数组tar(存储图像输入trt)
// *代表变长数组, void表示就地更改不返回值
void normalization_and_flatten_cpu(cv::Mat &src, float *mean, float *std, float* tar);

// 定义resize函数,返回cvmat
// 输入包括 已经读取的图片, 目标的宽高, 以及插值策略
// 用&传递src的原因: 直接传值的化会复制一个新的副本,
// 而cvmat一般都很大, 会影响处理速度, 而&只会传递对象的引用, 不会产生新数据
// const的原因: tarH tarW在整个流程中不应该被改变,因此声明为常量保证不出问题
// &则没有必要,但是如果后续更换成其他数据类型,&更有兼容性
cv::Mat preprocess_resize_cpu(cv::Mat &src, const int& tarH, const int& tarW, tactics tac);

// 整体前处理流程: BGR2RGB Normalization resize
// 输入参数图片 目标的宽高 ret:处理后的一维图像 均值mean 标准差std , 这里访问数组我用的是迭代器方法,下面gpu用的是index法
// *代表变长数组
void preprocess_cpu(cv::Mat &src, float* ret ,const int& tarH, const int& tarW, float* mean, float* std, tactics tac);


// 接下来是利用gpu进行前处理, 在gpu上对矩阵进行运算和传值
void preprocess_resize_cvt_norm_trans_gpu(cv::Mat &h_src, float* d_tar, const int& tarH, const int& tarW, float* mean, float* std, tactics tac);
void resize_bilinear_gpu(float *d_tar, uint8_t *d_src, int tarW, int tarH, int srcW, int srcH, float *mean, float *std, tactics tac);

} //namespace preprocess



#endif