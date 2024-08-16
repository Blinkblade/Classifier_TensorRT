#include "trt_preprocess.hpp"
#include "cuda_runtime_api.h"
#include <cmath>



namespace preprocess{



// 声明
TransInfo    trans;
AffineMatrix affine_matrix;

// 初始化仿射变换矩阵
void warpaffine_init(int srcH, int srcW, int tarH, int tarW){
    // extern  TransInfo    trans;
    trans.src_h = srcH;
    trans.src_w = srcW;
    trans.tar_h = tarH;
    trans.tar_w = tarW;
    // 再trt_preprocess hpp 中 声明了, 在这个函数中初始化
    preprocess::affine_matrix.init(trans);
}

__host__ __device__ void affine_transformation(
    float trans_matrix[6], int src_x, int src_y, 
    float* tar_x, float* tar_y)
{
    // 为输入的点计算新位置
    *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
    *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
    
}



// 最邻近插值值直接resize, 变换通道/形状/norm可以一起进行
__global__ void nearest_BGR2RGB_nhwc2nchw_norm_kernel(
    float* tar, uint8_t* src,
    int tarW, int tarH,
    int srcW, int srcH,
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std
)
{
    //首先使用xy定位处理位置,很经典的二维定位法
    // blockidx.x索引范围是[0,tarW/32+1(griddim)], blockDim.x值是32, threadIdx.x索引范围是[0,32]
    // 相当于每32个thread进一个block,索引范围正好涵盖: tarW + 1
    // 核函数运行时,这里就是 某个确定的block中的某个确定线程,其将会对应上图像上的固定位置
    // 也就是说,x代表的实际意义就是第blockIdx个block上的第threadIdx个线程,应该计算图像的哪个位置?
    // 由于索引范围完全覆盖图像,因此可以并行完成图像所有位置的计算
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 最邻近插值resize思路很简单,直接将xy还原到src上取得最近的像素值
    // 因此,我们直接拿到当前xy再src对应的坐标,floor就是四舍五入
    // x*w -> x对应h， y*h -> y对应w
    int src_x = floor((float)x*scaled_w);
    int src_y = floor((float)y*scaled_h);

    // 如果坐标超出范围就不处理
    if(src_x < 0 || src_x > srcW || src_y < 0 || src_y > srcH){

    }
    else{
        // 再cuda和trt计算中二维图像也采用一维的存储,因此需要组织index来实现二维访问
        // 对于xy再一维中的index就是行号*列数 + 列号
        // y代表行x代表列在cuda中比较常见,因为cuda block grid的xyz也是这样
        int tarIdx = y * tarW + x;
        // resize在取得srcxy的那一刻就已经完成了,接下来对这个位置的像素进行处理即可
        // 计算图像面积,用于组织数据到nchw的格式
        int imageArea = tarH * tarW;
        // src.data的组织形式是: pos0(b,g,r), pos1(b,g,r)……
        // 因此访问到对应位置需要*3
        int srcIdx = (src_y * srcW + src_x) * 3;

        // 接下来直接实现对应位置的处理和赋值,bgr->rgb,赋值顺序要反过来
        // 每个通道的像素进行各自的norm,norm的顺序是按src来的
        // red channel, tar:rgb,src:bgr
        tar[imageArea*0 + tarIdx] = ( src[srcIdx + 2]/255.f - d_mean[2] ) / d_std[2];
        // grenn channel
        tar[imageArea*1 + tarIdx] = ( src[srcIdx + 1]/255.f - d_mean[1] ) / d_std[1];
        // blue channel
        tar[imageArea*2 + tarIdx] = ( src[srcIdx + 0]/255.f - d_mean[0] ) / d_std[0];

    }    


};


__global__ void bilinear_BGR2RGB_nhwc2nchw_norm_kernel(
    float* tar, uint8_t* src,
    int tarW, int tarH,
    int srcW, int srcH,
    int scale_w, int scale_h,
    float* d_mean, float* d_std 
)
{
    // 同样先取得当前线程需要计算的xy
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 同样,还是需要通过scale定位到src位置
    // 这次我们找左上角,然后用+1定位其他三个
    // 这里的0.5指的是: x y 坐标代表的时像素的左上角,而我们认为像素的值应该是中心值
    // 即(x+0.5,y+0.5),我们希望找到中心值在原图上的位置
    // -0.5代表,定位到中心值之后回归左上角的传统坐标系中,这样才能定位到中心值周围的四个座标点
    int src_y1 = floor((y + 0.5) * scale_h - 0.5);
    int src_x1 = floor((x + 0.5) * scale_w - 0.5);
    // +1定位右下角，这样2 2组合就可以表示四个点
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    }
    else{

        // 先计算taridx,一维组织形式
        int tarIdx    = y * tarW  + x;
        int tarArea   = tarW * tarH;

        // xy 映射到原图的真实坐标
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = ((y + 0.5) * scale_h - 0.5) - src_y1;
        float tw   = ((x + 0.5) * scale_w - 0.5) - src_x1;

        // 计算五个坐标的四个区域的面积
        // 即th tw在四个坐标中间,分割出的四个矩形
        float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
        float a1_2 = tw * (1.0 - th);          //左下
        float a2_1 = (1.0 - tw) * th;          //右上
        float a2_2 = tw * th;                  //左上

        // 同样计算周围四个坐标的src索引,因为我们要取到对应位置的值用于插值计算
        // src.data的组织形式是: pos0(b,g,r), pos1(b,g,r)……
        // 因此访问到对应位置需要*3
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下
        // 接下来插值/分组/norm/赋值
        // 接下来直接实现对应位置的处理和赋值,bgr->rgb,赋值顺序要反过来
        // 每个通道的像素进行各自的norm,norm的顺序是按src来的
        // red channel, tar:rgb,src:bgr
        tar[tarArea * 0 + tarIdx] = (round((  //左上*右下
                                            src[srcIdx1_1 + 2] * a1_1 + 
                                            //右上*左下
                                            src[srcIdx1_2 + 2] * a1_2 +
                                            //左下*右上
                                            src[srcIdx2_1 + 2] * a2_1 +
                                            //右下*左上
                                            src[srcIdx2_2 + 2] * a2_2))/255.f - d_mean[2]) / d_std[2];
        // grenn channel
        tar[tarArea * 1 + tarIdx] = (round((  //左上*右下
                                            src[srcIdx1_1 + 1] * a1_1 + 
                                            //右上*左下
                                            src[srcIdx1_2 + 1] * a1_2 +
                                            //左下*右上
                                            src[srcIdx2_1 + 1] * a2_1 +
                                            //右下*左上
                                            src[srcIdx2_2 + 1] * a2_2))/255.f - d_mean[1]) / d_std[1];
        // blue channel, tar:rgb,src:bgr
        tar[tarArea * 2 + tarIdx] = (round((  //左上*右下
                                            src[srcIdx1_1 + 0] * a1_1 + 
                                            //右上*左下
                                            src[srcIdx1_2 + 0] * a1_2 +
                                            //左下*右上
                                            src[srcIdx2_1 + 0] * a2_1 +
                                            //右下*左上
                                            src[srcIdx2_2 + 0] * a2_2))/255.f - d_mean[0]) / d_std[0];

    }


};

// 这里主要是保持原图缩放比,其他和上面一样
__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std) 
{
    // resized之后的图tar上的坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
        float th   = (float)y * scaled_h - src_y1;
        float tw   = (float)x * scaled_w - src_x1;

        // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
        float a1_1 = (1.0 - tw) * (1.0 - th);  // 右下
        float a1_2 = tw * (1.0 - th);          // 左下
        float a2_1 = (1.0 - tw) * th;          // 右上
        float a2_2 = tw * th;                  // 左上

        // bilinear interpolation -- 计算4个坐标所对应的索引
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  // 左上
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  // 右上
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  // 左下
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  // 右下

        // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
        y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
        x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

        // bilinear interpolation -- 计算resized之后的图的索引
        int tarIdx    = (y * tarW  + x) * 3;
        int tarArea   = tarW * tarH;

        // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift + nhwc2nchw
        tar[tarIdx + tarArea * 0] = 
            (round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f - d_mean[2]) / d_std[2];

        tar[tarIdx + tarArea * 1] = 
            (round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f - d_mean[1]) / d_std[1];

        tar[tarIdx + tarArea * 2] = 
            (round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f - d_mean[0]) / d_std[0];
    }
}


// 仿射变换+RGB2BGR+nhwc2nchw+Normalization
__global__ void affine_BGR2RGB_nhwc2nchw_norm_kernel(
    float* tar, uint8_t* src, 
    AffineMatrix affine_matrix,
    // srcH,srcW,tarH,tarW被存储再trans中
    TransInfo trans,
    float* d_mean, float* d_std
    )
{
    // 根据线程取得定位到当前xy
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // 先确定当前xy对应原图的位置
    float src_x, src_y;
    // 从 tar 映射回 src, 属于反变换
    affine_transformation(affine_matrix.reverse, x+0.5, y+0.5, &src_x, &src_y);

    // 为tar[x][y]取得对应的值,包括rgb2bgr, 维度变换, norm
    // 取定双线性插值的两个角点
    int src_x1 = floor(src_x - 0.5);
    int src_y1 = floor(src_y - 0.5);
    int src_x2 = src_x1 + 1;
    int src_y2 = src_y1 + 1;

    // 对应源去值错误,则没有值
    if (src_y1 < 0 || src_x1 < 0 || src_y1 > trans.src_h || src_x1 > trans.src_w) {
    } 
    else{
        // src point对应左上角点的长宽
        float tw   = src_x - src_x1;
        float th   = src_y - src_y1;

        // 计算四个部分的面积
        float a1_1 = (1-tw) * (1-th); //右下
        float a1_2 = tw * (1-th); // 左下
        float a2_1 = (1-tw) * th; //右上
        float a2_2 = tw * th; //左上

        // 索引到源的角点index, *3是因为每个位置有三个像素, *3可以索引到第一个像素(hwc格式存储)   
        int srcIdx1_1 = (src_y1 * trans.src_w + src_x1) * 3; //左上
        int srcIdx1_2 = (src_y1 * trans.src_w + src_x2) * 3; //右上
        int srcIdx2_1 = (src_y2 * trans.src_w + src_x1) * 3; //左下
        int srcIdx2_2 = (src_y2 * trans.src_w + src_x2) * 3; //右下

        // 计算x y对应tar的idx(hw位置),
        int tarIdx = y * trans.tar_w + x;
        int tarArea = trans.tar_w * trans.tar_h;

        // tar中按照nchw格式存储,先索引c再索引hw, 并且通道排列是rgb
        // bgr2rgb, 通道idx应该是 0 1 2 = 2 1 0
        // 通道0,r
        tar[0*tarArea + tarIdx] = 
            (round((a1_1 * src[srcIdx1_1 + 2] + 
                    a1_2 * src[srcIdx1_2 + 2] +
                    a2_1 * src[srcIdx2_1 + 2] +
                    a2_2 * src[srcIdx2_2 + 2])) / 255.0f - d_mean[2]) / d_std[2];


        // 通道1,g
        tar[1*tarArea + tarIdx] = 
            (round((a1_1 * src[srcIdx1_1 + 1] + 
                    a1_2 * src[srcIdx1_2 + 1] +
                    a2_1 * src[srcIdx2_1 + 1] +
                    a2_2 * src[srcIdx2_2 + 1])) / 255.0f - d_mean[1]) / d_std[1];
        // 通道2,b
        tar[2*tarArea + tarIdx] = 
            (round((a1_1 * src[srcIdx1_1 + 0] + 
                    a1_2 * src[srcIdx1_2 + 0] +
                    a2_1 * src[srcIdx2_1 + 0] +
                    a2_2 * src[srcIdx2_2 + 0])) / 255.0f - d_mean[0]) / d_std[0];

    }

}



// 这里主要进行一下核函数的配置,分配block grid等等
void resize_bilinear_gpu(float *d_tar, uint8_t *d_src, int tarW, int tarH, int srcW, int srcH, float *d_mean, float *d_std, tactics tac)
{
    // 由于操作对象是二维图像, 因此 逻辑划分也是二维
    // 每个Block限制为1024个线程, 因此设置满是合理的选择
    // 32 * 32 = 1024, 这是因为 NVIDIA GPU 的 "warp"（一次性执行的线程集合）大小是 32。
    // 选择 32x32 使线程块的总大小等于 32 的倍数，可以更好地利用 GPU 的硬件并行能力。 
    // 不知道如何选择block grid dim时, 可以先从32开始尝试
    dim3 dimBlock(32, 32, 1);
    // 要保证线程可以完全覆盖图像的每一个位置,
    // 需要grid block的维度乘起来与tarw*tarh相等
    // +1是为了防止bank conflict
    dim3 dimGrid(tarW/32, tarH/32, 1);

    // 设置resize所需的变换尺度
    // 注意,这里一定是(float)而不是float(),因为后者实际上是整形运算
    float scaled_h = (float)srcH/tarH ;
    float scaled_w = (float)srcW/tarW ;
    // 谁大选谁
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    switch (tac) {
    case preprocess::tactics::GPU_NEAREST:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
        break;
    case preprocess::tactics::GPU_NEAREST_CENTER:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>>
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
        break;
    case preprocess::tactics::GPU_BILINEAR:
        bilinear_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean, d_std);
        break;
    case preprocess::tactics::GPU_BILINEAR_CENTER:
        bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel 
                <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
        break;
    
    case preprocess::tactics::GPU_WARP_AFFINE:
    // 初始化仿射矩阵然后输入到核函数
        warpaffine_init(srcH, srcW, tarH, tarW);
        affine_BGR2RGB_nhwc2nchw_norm_kernel
                <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, affine_matrix, trans, d_mean, d_std);
        break;

    default:
        LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
        exit(1);
    }


};



}//preprocess

