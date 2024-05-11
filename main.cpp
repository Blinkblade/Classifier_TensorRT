#include <iostream>

#include "opencv2/opencv.hpp"

#include "utils.hpp"
#include "trt_logger.hpp"
#include "trt_timer.hpp"
#include "trt_model.hpp"
#include "trt_preprocess.hpp"
#include "test_cuda.hpp"
#include "trt_classifier.hpp"
#include "trt_worker.hpp"

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;


// 不使用整个std命名空间,可以指定部分常用且不易冲突的命名
using std::cout;
using std::endl;


// 这是一些测试函数,用于正式执行之前测试各个功能模块有没有问题
// 提供一种思路,这里正式使用的时候没有用可以直接注释掉
// 测试timer是否正常工作
void timer_test(){
    std::string msg = "test_timer";
    // volatile防止编译器优化掉这个无用参数
    volatile int dummy = 0;
    timer::Timer tmr = timer::Timer();
    tmr.start_cpu();
    for(int i = 0; i < 10000; i++){
        dummy+=1;
    };
    tmr.stop_cpu();
    // 输入的字符串相当于事件名称,指定us为单位
    tmr.duration_cpu<timer::Timer::us>(msg);
    cout << "timer_test 测试完毕" <<endl;
}

// 测试logger是否正常工作
void logger_test(logger::Logger logger){
    logger::Logger::Severity s_test = logger::Logger::Severity::kWARNING;
    logger.log(s_test,"s_test4564566!!!");
    cout << "logger_test 测试完毕" <<endl;
}

// 测试cpuresize是否正确
void resize_cpu_test(){
    // 读入图片
    // 路径是按照build文件夹设置的
    std::string image_path = "data/cat.png";
    cv::Mat src = cv::imread(image_path);
    // cout << "src type : "<< src.type() <<endl;
    // 最邻近插值resize
    std::string nearest_save_path = "data/cat_nearest.png";
    cv::Mat tar = preprocess::preprocess_resize_cpu(src, 400, 800, preprocess::tactics::CPU_NEAREST);
    cv::imwrite(nearest_save_path, tar);
    cout << "nearest resize 保存至 " << nearest_save_path << endl;

    // 双线性插值
    std::string linear_save_path = "data/cat_linear.png";
    tar = preprocess::preprocess_resize_cpu(src, 400, 800, preprocess::tactics::CPU_BILINEAR);
    cv::imwrite(linear_save_path, tar);
    cout << "linear resize 保存至 " << linear_save_path << endl;
    cout << "resize_cpu_test 测试完毕" <<endl;
}

// 测试cpu preprocess是否正确
void preprocess_cpu_test(){
    // 读入图片
    std::string image_path = "data/cat.png";
    cv::Mat src = cv::imread(image_path);
    // 分配空间,图像大小: h w c
    float* ret =(float *)malloc(sizeof(float) * src.rows * src.cols * 3);
    // imagenet的均值和标准差
    float mean[]       = {0.485, 0.456, 0.406};
    float std[]        = {0.229, 0.224, 0.225};
    
    preprocess::preprocess_cpu(src, ret, 400, 800, mean, std, preprocess::tactics::CPU_NEAREST);
    cout << "转换后一维图像某个值 "<< ret[500] <<endl; 
    cout << "prprocess_cpu_test 测试完毕" <<endl;

}

// 测试cpu preprocess是否正确
void getEnginePath_test(){
    // 设置onnx
    std::string onnxPath = "models/onnx/resnet34.onnx";
    std::string enginePath = getEnginePath(onnxPath);
    cout << "getEnginePath_test 测试完毕" <<endl;

}

// 测试cpu上从构建加载到推理的流程
void classifier_test(std::string onnxPath, logger::Level level, model::Params params){

    // 指定到cpu上
    params.dev = model::device::CPU;    
    // 初始化一个model,用智能指针
    auto classifier = model::classifier::make_classifier(onnxPath, level, params);
    // 推理测试
    classifier->load_image("data/cat.png");
    classifier->inference();

    classifier->load_image("data/eagle.png");
    classifier->inference();

    classifier->load_image("data/fox.png");
    classifier->inference();
}

// 测试cpu上从构建加载到推理的流程
void classifier_test_gpu(std::string onnxPath, logger::Level level, model::Params params){
    // 转换到gpu上
    params.dev = model::device::GPU;
    // 初始化一个model,用智能指针
    auto classifier = model::classifier::make_classifier(onnxPath, level, params);
    // 推理测试
    classifier->load_image("data/cat.png");
    classifier->inference();

    classifier->load_image("data/eagle.png");
    classifier->inference();

    classifier->load_image("data/fox.png");
    classifier->inference();
}


// 对文件夹中所有图像进行推理
void process_images_in_directory(const fs::path& directory_path, std::shared_ptr<worker::Worker> worker) {
    
    // 遍历给定目录
    if (fs::exists(directory_path) && fs::is_directory(directory_path)) {
        fs::directory_iterator end_iter; // 默认构造产生的迭代器作为终点
        for (fs::directory_iterator dir_itr(directory_path); dir_itr != end_iter; ++dir_itr) {
            if (fs::is_regular_file(dir_itr->status())) {
                fs::path current_file = dir_itr->path();
                if (current_file.extension() == ".png" || current_file.extension() == ".jpg" || current_file.extension() == ".jpeg") {
                    std::cout << "Processing: " << current_file << std::endl;
                    worker->inference(current_file.string()); // 执行推理
                }
            }
        }
    }
}




// 外部传参
int main(int argc, char const *argv[])
{
    // 设置全局日志等级,只有小于等于该日志等级的才会输出
    // debug是最高等级也就是最详细的日志
    logger::Level level = logger::Level::DEBUG;
    // 初始化日志变量
    logger::Logger logger(level);
    // 初始化onnxpath
    std::string onnxPath = "models/onnx/resnet34.onnx";
    // 创建一个模型参数对象
    auto params = model::Params();
    // 设置输入图像的形状（224x224，3个通道）
    params.img_info = {224, 224, 3};
    // 设置分类任务的类别数量
    params.numclass = 1000;
    // 设置任务类型为图像分类
    params.task = model::task_type::CLASSIFICATION;
    // 设置设备类型为 GPU
    params.dev = model::device::GPU;


    // 单帧推理
    auto worker = worker::create_worker(onnxPath, level, params);
    worker->inference("data/cat.png");

    // 设置文件夹
    fs::path directory = "data";
    cout << "============================== 文件夹推理 ==============================" << endl;
    process_images_in_directory(directory, worker);

    cout << "main 执行完毕!" <<endl;

    return 0;
}

