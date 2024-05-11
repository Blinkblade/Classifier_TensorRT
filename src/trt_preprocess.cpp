#include "trt_preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "cuda_runtime_api.h"

namespace preprocess{

// 输入参数图片 均值mean 标准差std , 输出的一维数组
// *代表变长数组, void表示就地更改不返回值
// tar是一个一维数组,因为trt引擎只接受chw格式的一维输入
void normalization_and_flatten_cpu(cv::Mat &src, float *mean, float *std, float* tar)
{   
    // 先计算图像的大小,用于划分tar的存储位置
    int src_Area = src.rows * src.cols;
    // normalization : 每个通道减去各自的均值然后除以标准差
    // 使用迭代器访问每个位置, 并对三个通道依次处理
    // 每个位置是一个三通道的像素,Vec3b代表8UC3
    cv::MatIterator_<cv::Vec3b> it,end;
    // 很经典的迭代器写法,迭代器是一种指针, begin()指向第一个元素, end()指向最后一个元素的下一个元素
    int i = 0;
    for(it = src.begin<cv::Vec3b>(), end = src.end<cv::Vec3b>(); it != end; ++it){
        // 初始化一个3d vector接收迭代器的值
        // 注意使用引用这样才能修改像素值
        // 实际上这个pixel没有用,值是为了验证处理过程,真正的结果值存储在tar中,因此真正运行时可以注释掉
        // 如果需要测试就取消下面的注释然后注释掉tar三行
        cv::Vec3b& pixel = *it;
        // // 考虑到到这一步已经实现bgr2rgb因此0 1 2分别是r g b
        // // 图像默认编码时8UC3, 也就是uchar类型0~255,因此先/255归一化
        // pixel[0] = (pixel[0]/ 255.0f  - mean[0])/std[0];
        // pixel[1] = (pixel[1]/ 255.0f  - mean[1])/std[1];
        // pixel[2] = (pixel[2]/ 255.0f  - mean[2])/std[2];

        // 前src_Area是R,中间src_Area是G,最后src_Area是B
        // 测试时norm时可以注释
        tar[i + 0*src_Area] = (pixel[0]/ 255.0f  - mean[0])/std[0];
        tar[i + 1*src_Area] = (pixel[1]/ 255.0f  - mean[1])/std[1];
        tar[i + 2*src_Area] = (pixel[2]/ 255.0f  - mean[2])/std[2];
        i++;
    }
    
}

// 定义resize函数,返回cvmat
// 输入包括 已经读取的图片(&类型对原图操作), 目标的宽高, 以及插值策略
// 用&传递src的原因: 直接传值的化会复制一个新的副本,
// 而cvmat一般都很大, 会影响处理速度, 而&只会传递对象的引用, 不会产生新数据
// const的原因: tarH tarW在整个流程中不应该被改变,因此声明为常量保证不出问题
// &则没有必要,但是如果后续更换成其他数据类型,&更有兼容性
cv::Mat preprocess_resize_cpu(cv::Mat &src,const int& tarH,const int& tarW, tactics tac){
    // 生成一个tar用于返回
    cv::Mat tar;
    // 用其他变量接收tarH,tarW: 可更改,拓展性强
    int resizeH = tarH;
    int resizeW = tarW;

    // 不同策略riseze
    switch (tac)
    {
    // 最近插值
    case tactics::CPU_NEAREST :
    // 源图像, 目标图像, dst_size, 水平/垂直缩放因子(0代表不使用), 最近插值
    // SIZE是先width 后 height
        cv::resize(src, tar, cv::Size(resizeW,resizeH), 0, 0, cv::INTER_NEAREST);
        break;

    case tactics::CPU_BILINEAR :

        cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
        break;
    // 都不是就报错
    default:
        LOGE("Wrong CPU Resize Tactics! please check for =preprocess_resize_cpu= !");
        break;
    }
    return tar;
};

// 整体流程
// resize -> BGR2RGB -> Normalization -> nhwc2nchw
// 输入参数读图片 ret:处理后的一维图像 目标的宽高 均值mean 标准差std 
// *代表变长数组
void preprocess_cpu(cv::Mat &src, float* ret, const int &tarH, const int &tarW, float *mean, float *std, tactics tac)
{
    cv::Mat tar;
    // resize
    tar = preprocess_resize_cpu(src, tarH, tarW, tac);
    // BGR2RGB,调用cv自带的函数完成, 输入分别是input mat和output mat
    cv::cvtColor(tar, tar, cv::COLOR_BGR2RGB);
    // normalization : 每个通道减去各自的均值然后除以标准差
    // flatten whc的cvmat按照 chw展平成1D的float向量以便输入到trt推理引擎中
    normalization_and_flatten_cpu(tar, mean, std, ret);
    
}
// 这里主要进行一些初始化包括空间分配和路径检查等,主要的运算通过cuda进行
void preprocess_resize_cvt_norm_trans_gpu(cv::Mat &h_src, float *d_tar, const int &tarH, const int &tarW, float *mean, float *std, tactics tac) {

    // 先设置好动态数组等待malloc分配
    // gpu上的mean std
    float* d_mean = nullptr;
    float* d_std = nullptr;
    // gpu image 
    // cvmat默认是8UC3, unint8_t和unsigned char是一个东西
    uint8_t* d_src  = nullptr;
    // 计算shape
    // 行数为height, 列数为widht
    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;
    // 3个通道各自一个值
    int norm_size = chan * sizeof(float);
    // 图像大小
    int src_size = height * width * chan * sizeof(uint8_t);

    // 为指针分配cuda空间
    CUDA_CHECK(cudaMalloc(&d_mean, norm_size));
    CUDA_CHECK(cudaMalloc(&d_std, norm_size));
    CUDA_CHECK(cudaMalloc(&d_src, src_size));

    // 向对应空间传值
    CUDA_CHECK(cudaMemcpy(d_mean, mean, norm_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, std, norm_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    // 把图像数据cpy进去
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

    preprocess::resize_bilinear_gpu(d_tar, d_src, tarW, tarH, width, height, d_mean, d_std, tac);

    // 结束之前先同步
    CUDA_CHECK(cudaDeviceSynchronize());
    // 处理完之后释放空间爱你,释放顺序要对称
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_std));
    CUDA_CHECK(cudaFree(d_mean));




};
}