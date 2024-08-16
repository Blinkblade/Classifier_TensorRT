#include "trt_detector.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.hpp"
#include "trt_logger.hpp"
#include <vector>
#include "coco_labels.hpp"





namespace model{
namespace detector{


// 计算两个box的
float iou_calc(bbox bbox1, bbox bbox2){
    // 计算交界点坐标
    auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
    auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
    auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
    auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

    // 交界面积
    float inter_w = inter_x1 - inter_x0;
    float inter_h = inter_y1 - inter_y0;
    
    float inter_area = inter_w * inter_h;
    float union_area = 
        (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) + 
        (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - 
        inter_area;
    
    return inter_area / union_area;
}




// 1.创建engine和context
// 2.分配bindings和memory,根据模型不同设置的会不同
// 3.初始化一些成员变量
void Detector::setup(void const *data, std::size_t size)
{
    // 创建上下文的流程为: 创建runtime -> runtime 反序列化 生成 engine -> engine 生成上下文 context
    // 使用create runtime创建IRuntime指针
    this->m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*this->m_logger), destroy_trt_ptr<nvinfer1::IRuntime>);
    // runtime反序列化生成engine, data为传入的序列化engine,size为数据大小
    this->m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(this->m_runtime->deserializeCudaEngine(data,size), destroy_trt_ptr<nvinfer1::ICudaEngine>);
    // 从engine生成用于推理的上下文context
    this->m_context = std::shared_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext(), destroy_trt_ptr<nvinfer1::IExecutionContext>);
    LOGD("===============Detector createExecutionContext SUCCESSFULLY===============");


    // 分配线程
    CUDA_CHECK(cudaStreamCreate(&this->m_stream));

    // 接下来计算并分配memory然后为Binds分配内存
    // 首先拿到input 和 output形状, 可以直接从模型的输入输出中读取形状
    // binding的结构[[list of inputs],[list of outputs]]
    // binding是从onnx转换到trt的模型结构中读取出来的
    // output shape : float32[1,8400,84]
    this->m_inputDims = this->m_context->getBindingDimensions(0);
    this->m_outputDims = this->m_context->getBindingDimensions(1);
    // 考虑到2d detector model基本都是1 input, 1 output, 这边这么写。如果像BEVFusion这种有多输出的需要修改
    

    // 计算输入输出大小
    this->m_inputSize = this->mParams->img_info.h * this->mParams->img_info.w * this->mParams->img_info.c * sizeof(float);
    this->m_imgArea = this->mParams->img_info.h * this->mParams->img_info.w;
    this->m_outputSize = this->m_outputDims.d[1] * this->m_outputDims.d[2] * sizeof(float);

    // 分配memory和binds
    // 0代表cpu 1 代表gpu
    CUDA_CHECK(cudaMallocHost(&this->m_inputMemory[0], this->m_inputSize));
    CUDA_CHECK(cudaMallocHost(&this->m_outputMemory[0], this->m_outputSize));
    CUDA_CHECK(cudaMalloc(&this->m_inputMemory[1], this->m_inputSize));
    CUDA_CHECK(cudaMalloc(&this->m_outputMemory[1], this->m_outputSize));

    // 将bind绑定到对应memory中, 0 是输入, 1 是输出
    // bindings中index是按照输入和输出顺序排列的
    this->m_bindings[0] = this->m_inputMemory[1];
    this->m_bindings[1] = this->m_outputMemory[1];
}

// 重置detector
void Detector::reset_task()
{
    // 清空当前bbox
    this->m_bboxes.clear();
}


bool Detector::preprocess_cpu()
{
    // 均值和标准差
    // yolov8的默认均值和标准差设置是0和1,相当于不起作用
    float mean[]       = {0, 0, 0};
    float std[]        = {1, 1, 1};

    // 从图像路径中读取图像并存储到成员变量中
    LOG("Detector Inference Load Image from : %s", this->m_imagePath.c_str());
    this->m_inputImage = cv::imread(this->m_imagePath);
    if (this->m_inputImage.data == nullptr) {
        LOGE("ERROR: Detector Image file not founded! Detector Program terminated"); 
        return false;
    }

    //测速
    this->m_timer->start_cpu();

    // 取得输入模型的图像大小
    int tarH = this->mParams->img_info.h;
    int tarW = this->mParams->img_info.w;

    // 将读取的图像resize到模型输入大小
    // src, dst, size(w,h), fx, fy , 插值方式
    cv::resize(this->m_inputImage,this->m_inputImage,cv::Size(tarW,tarH),0,0,cv::INTER_LINEAR);

    // 包括bgr2rgb,mean std以及NHWC->NCHW
    // 索引
    int index;
    // 三个通道的起始位置
    int offsetCH0 = this->m_imgArea * 0;
    int offsetCH1 = this->m_imgArea * 1;
    int offsetCH2 = this->m_imgArea * 2;

    // 取得nchw用于生成index
    // int N = this->m_inputDims.d[0]; // 其实没用
    int C = this->m_inputDims.d[1];
    int H = this->m_inputDims.d[2];
    int W = this->m_inputDims.d[3];

    // 主要前处理流程
    // 推理引擎存储的input格式是nchw(因为是onnx导出然后转换而来的)
    // 而opencv的图像data的排列是nhwc,因此对应的
    // 0->n,1->c,2->h,3->w
    for(int i = 0; i < this->m_inputDims.d[2]; i++){
        for(int j = 0; j < this->m_inputDims.d[3]; j++){
            // 解算当前input对应的cv image 的 index
            // 索引到cv image的hw位置,接下来有三个index的通道
            index = i*C*W + j*C;
            // 三个通道分别赋值 左:RGB, 右:BGR, /255f归一化,并计算mean std
            this->m_inputMemory[0][offsetCH0++] = (this->m_inputImage.data[index + 2]/255.0f - mean[0])/std[0];
            this->m_inputMemory[0][offsetCH1++] = (this->m_inputImage.data[index + 1]/255.0f - mean[1])/std[1];
            this->m_inputMemory[0][offsetCH2++] = (this->m_inputImage.data[index + 0]/255.0f - mean[2])/std[2];
        }
    }

    // cpu处理完数据后,将数据迁移到cuda上
    CUDA_CHECK(cudaMemcpyAsync(this->m_inputMemory[1], this->m_inputMemory[0], this->m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, this->m_stream));

    // 结束计时
    this->m_timer->stop_cpu();
    this->m_timer->duration_cpu<timer::Timer::ms>("Detector preprocess(CPU)");
    return true;

}

bool Detector::preprocess_gpu()
{
    // 均值和标准差
    // yolov8的默认均值和标准差设置是0和1,相当于不起作用
    float mean[]       = {0, 0, 0};
    float std[]        = {1, 1, 1};

    // 从图像路径中读取图像并存储到成员变量中
    LOG("Detector Inference Load Image from : %s", this->m_imagePath.c_str());
    this->m_inputImage = cv::imread(this->m_imagePath);
    if (this->m_inputImage.data == nullptr) {
        LOGE("ERROR: Detector Image file not founded! Detector Program terminated"); 
        return false;
    }


    // GPU测速
    this->m_timer->start_gpu();

    int tarH = this->mParams->img_info.h;
    int tarW = this->mParams->img_info.w;

    preprocess::preprocess_resize_cvt_norm_trans_gpu(this->m_inputImage, this->m_inputMemory[1], 
                                                    tarH, tarW, mean, std, 
                                                    preprocess::tactics::GPU_WARP_AFFINE);


    //测速
    this->m_timer->stop_gpu();
    this->m_timer->duration_gpu("Detector  preprocess(GPU)");


    
    return true;
}

// 
bool Detector::postprocess_cpu()
{
    m_timer->start_cpu();
    // 从output中取出推理结果,将其放到cpu上
    // 计算输出向量大小, output dim 是从 模型结构直接读取的和Onnx差不多
    // outputshape float32[1,8400,84] batchsize proposals channel
    int output_size = this->m_outputDims.d[1] * this->m_outputDims.d[2] * sizeof(float);
    // 将cuda上的output转移到host上来
    CUDA_CHECK(cudaMemcpyAsync(this->m_outputMemory[0], this->m_outputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, this->m_stream));
    // 同步
    CUDA_CHECK(cudaStreamSynchronize(this->m_stream));

    /*Postprocess -- yolov8的postprocess需要做的事情*/
    /*
     * 1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
     * 2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
     * 3. 把最终得到的bbox绘制到原图中
     */

    float conf_threshold = 0.25; //用来过滤decode时的bboxes
    float nms_threshold  = 0.45;  //用来过滤nms时的bboxes

    /*Postprocess -- 1. decode*/
    /*
     * 我们需要做的就是将[batch, proposals, channel]转换为vector<bbox>
     * 几个步骤:
     * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height
     * 2. 对每一个bbox中对应的ch中，找到最大的class label, 可以使用std::max_element
     * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
     * 4. 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine matrix来进行逆变换)
     * 5. 将转换好的x0, y0, x1, y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
     */
    // 8400
    int boxes_count = this->m_outputDims.d[1];
    // 每个位置的通道数目
    int channels = this->m_outputDims.d[2];
    // 84 (4 + 80), 减去cx cy w h四个维度
    int class_count = channels - 4;
    // 取得具体的特征用于解码
    float* tensor;

    // 存储解码后的相关属性
    float cx, cy, w, h, obj, prob, conf;
    // 从cx cy wh 转换为角点坐标
    float x0, y0, x1, y1;
    // 标签(class)
    int label;

    // 取得每个位置的tensor,然后对其解码
    for(int i = 0; i < boxes_count; i++){
        // 1 * 8400 * 84 的连续一维向量, 按照(84, 84, 84,……)的格式存储
        tensor = this->m_outputMemory[0] + i*channels;
        // 先判定置信度再考虑解码, 以label的值作为置信度
        // 计算最大位置距离class起始位置的距离作为Index
        label  = max_element(tensor + 4, tensor + 4 + class_count) - (tensor + 4);
        conf   = tensor[4 + label];
        if (conf < conf_threshold){
            // 置信度太低就直接跳过本轮
            continue;
        }
        // 从84个通道中拿到对应的值
        cx = tensor[0];
        cy = tensor[1];
        w = tensor[2];
        h = tensor[3];

        // 从cx cy中得到角点坐标
        x0 = cx - w/2;
        y0 = cy - h/2;
        x1 = x0 + w;
        y1 = y0 + h;
 
        // 使用逆变换可以快速将坐标映射回原图
        // affine matrix在preprocess时已经初始化过
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);

        // 此时已经得到了真正的角点坐标, 录入一个box
        detector::bbox yolo_box(x0, y0, x1, y1, conf, label);
        m_bboxes.emplace_back(yolo_box);
    }

    LOGD("the count of proposal bbox is %d", m_bboxes.size());

    // 初步取得proposal框之后, 需要进行nms
    /*Postprocess -- 2. NMS*/
    /* 
     * 几个步骤:
     * 1. 做一个IoU计算的lambda函数
     * 2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
     * 3. 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
     *    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU threshold
     */

    // 一次申请好空间, 减少空间变更次数
    vector<bbox> final_bboxes;
    final_bboxes.reserve(m_bboxes.size());
    // 按置信度排序, []就是一个及时的函数
    std::sort(m_bboxes.begin(), m_bboxes.end(), 
              [](bbox& box1, bbox& box2){return box1.confidence > box2.confidence;});

    // 优先挑选置信度更高的框
    for(int i = 0; i < m_bboxes.size(); i ++){
        // 已经被掠过的框直接跳过
        if (m_bboxes[i].flg_remove){
            continue;
        }
        // 由于框是按照置信度排序, 同位置同类别中, 置信度最大的一定在前面
        // 因此重叠时, 被排除的一定是后者
        final_bboxes.emplace_back(m_bboxes[i]);
        // 在后面找和当前同类别且重叠的框, 全部移除
        for (int j = i + 1; j < m_bboxes.size(); j ++) {
            if (m_bboxes[j].flg_remove){
                continue;
            }
            // 类别相同时, 判断是否重叠
            if (m_bboxes[i].label == m_bboxes[j].label){
                if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
                // 重叠时认为两个框预测的是同一个物体, 将置信度小的后者移除
                    m_bboxes[j].flg_remove = true;
            }
        }
    }
    // 到这里就已经得到了所有的预测框,
    LOGD("the count of bbox after NMS is %d", final_bboxes.size());
    // 绘图
    /*Postprocess -- draw_bbox*/
    /*
     * 几个步骤
     * 1. 通过label获取name
     * 2. 通过label获取color
     * 3. cv::rectangle
     * 4. cv::putText
     */
    string tag   = "detect-" + getPrec(this->mParams->prec);
    m_outputPath = changePath(m_imagePath, "../result", ".png", tag);

    int   font_face  = 0;
    float font_scale = 0.001 * MIN(m_inputImage.cols, m_inputImage.rows);
    int   font_thick = 2;
    int   baseline;
    CocoLabels labels;

    LOG("\tResult:");
    for (int i = 0; i < final_bboxes.size(); i ++){
        auto box = final_bboxes[i];
        auto name = labels.coco_get_label(box.label);
        auto rec_color = labels.coco_get_color(box.label);
        auto txt_color = labels.get_inverse_color(rec_color);
        auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
        auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thick, &baseline);

        int txt_height = txt_size.height + baseline + 10;
        int txt_width  = txt_size.width + 3;

        cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height - baseline + font_thick)));
        cv::Rect  txt_rec(round(box.x0 - font_thick), round(box.y0 - txt_height), txt_width, txt_height);
        cv::Rect  box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

        cv::rectangle(this->m_inputImage, box_rec, rec_color, 3);
        cv::rectangle(this->m_inputImage, txt_rec, rec_color, -1);
        cv::putText(this->m_inputImage, txt, txt_pos, font_face, font_scale, txt_color, font_thick, 16);

        LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
            name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);

    }
    LOG("\tSummary:");
    LOG("\t\tDetected Objects: %d", final_bboxes.size());
    LOG("");

    this->m_timer->stop_cpu();
    this->m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    cv::imwrite(this->m_outputPath, this->m_inputImage);
    LOG("\tsave image to %s\n", this->m_outputPath.c_str());

    return true;
}


bool Detector::postprocess_gpu() {
    return postprocess_cpu();
}


shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

} // namespace model
}   //namespace detector


