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
    // GPU仿射变换
    GPU_WARP_AFFINE     = 6,
};


// 仿射变换需要的信息,包括原图宽高和目标宽高
struct TransInfo{
    int src_w = 0;   // 源图像宽度
    int src_h = 0;   // 源图像高度
    int tar_w = 0;   // 目标图像宽度
    int tar_h = 0;   // 目标图像高度
    TransInfo() = default;  // 默认构造函数
    TransInfo(int srcW, int srcH, int tarW, int tarH):
        src_w(srcW), src_h(srcH), tar_w(tarW), tar_h(tarH){}  // 参数化构造函数，用于直接初始化成员变量
};


struct AffineMatrix{
    float forward[6];         // 正向变换矩阵数组
    float reverse[6];         // 反向变换矩阵数组
    float forward_scale;      // 正向变换的缩放比例
    float reverse_scale;      // 反向变换的缩放比例

    void calc_forward_matrix(TransInfo trans){
        // 设置矩阵第一行第一个元素为缩放因子
        // 缩放因子取目标wh/源wh较小的一个
        forward[0] = forward_scale;
        // 第一行第二个元素为0，表示在x方向上没有与y方向的交叉影响
        forward[1] = 0;
        // 第一行第三个元素为平移参数，计算方式是：将源图像宽度的一半乘以负缩放因子加上目标图像宽度的一半
        // 这样处理是为了将图像中心对齐到目标图像中心
        forward[2] = - forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
        // 设置矩阵第二行第一个元素为0，表示在y方向上没有与x方向的交叉影响
        forward[3] = 0;
        // 第二行第二个元素为缩放因子
        forward[4] = forward_scale;
        // 第二行第三个元素为平移参数，同样的逻辑，用于对齐y方向的图像中心
        forward[5] = - forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
    };

    void calc_reverse_matrix(TransInfo trans){
        // 设置矩阵第一行第一个元素为反向缩放因子
        reverse[0] = reverse_scale;
        // 第一行第二个元素为0，表示在x方向上没有与y方向的交叉影响
        reverse[1] = 0;
        // 第一行第三个元素为平移参数，计算方式是：将目标图像宽度的一半乘以负反向缩放因子加上源图像宽度的一半
        // 这样的处理是为了从目标图像回到源图像的位置，也是对齐图像中心
        reverse[2] = - reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
        // 设置矩阵第二行第一个元素为0，表示在y方向上没有与x方向的交叉影响
        reverse[3] = 0;
        // 第二行第二个元素为反向缩放因子
        reverse[4] = reverse_scale;
        // 第二行第三个元素为平移参数，用相似的逻辑处理y方向
        reverse[5] = - reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
    };


    void init(TransInfo trans){
        // 计算源图像和目标图像宽度的比例
        float scaled_w = (float)trans.tar_w / trans.src_w;
        // 计算源图像和目标图像高度的比例
        float scaled_h = (float)trans.tar_h / trans.src_h;
        // 取较小的比例作为正向变换的缩放因子，以确保整个图像都能适应在目标尺寸内
        forward_scale = (scaled_w < scaled_h ? scaled_w : scaled_h);
        // 反向缩放因子为正向缩放因子的倒数
        reverse_scale = 1 / forward_scale;

        calc_forward_matrix(trans); // 根据计算出的缩放因子计算正向变换矩阵
        calc_reverse_matrix(trans); // 根据计算出的缩放因子计算反向变换矩阵
    }
};


// 对结构体设置default instance
// extern代表它们在外部定义
extern  TransInfo    trans;
extern  AffineMatrix affine_matrix;

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


// 仿射变换
__host__ __device__ void affine_transformation(float trans_matrix[6], int src_x, int src_y, float* tar_x, float* tar_y);

} //namespace preprocess



#endif