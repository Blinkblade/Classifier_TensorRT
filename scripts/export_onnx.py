import torchvision
from torch import onnx
import os
import torch

# 获取torchvision model,默认是resnet34, 无指定权重
def get_model(model_name="resnet34", weights_path=None):
    # 获取 torchvision.models 模块
    models_module = torchvision.models
    # 动态获取模型modelname构造函数,没有就返回none
    model_constructor = getattr(models_module, model_name, None)
    if model_constructor is None:
        raise ValueError(f"Model {model_name} not found in torchvision.models")

    # 根据权重路径决定是否使用预训练权重
    if weights_path is None or not os.path.exists(weights_path):
        # 如果未提供权重路径或权重文件不存在，则使用预训练权重
        pretrained = True
        # 基于model name构造网络
        print("==========无有效权重路径, 创建 {} 模型,加载pytorch预训练权重==========".format(model_name))
        model = model_constructor(pretrained=pretrained)
    else:
        # 如果提供了权重路径且文件存在，加载指定的权重
        pretrained = False
        model = model_constructor(pretrained=pretrained)
        print("==========创建 {} 模型,从 {} 加载权重==========".format(model_name, weights_path))
        model.load_state_dict(torch.load(weights_path))

    return model

# 从pytorch导出onnx,输入模型和导出路径以及输入张量
def export_onnx(model:torch.nn.Module, onnx_path:str, inputs:torch.Tensor):
    # 迁移到gpu
    model.cuda()
    torch.onnx.export(
        model           =   model,
        # 输入是tuple,一定不能单输入
        args            =   (inputs,),
        f               =   onnx_path,
        # 输入输出名称都是列表
        input_names     =   ["input0"],
        output_names    =   ["output0"],
        opset_version   =   15,
    )

    print("========模型 {} 已导出onnx至 {} ==========".format(model._get_name(), onnx_path))



if __name__ == "__main__":

    model_name = "resnet34"
    weights_path = "weights/resnet34-b627a593.pth"
    # 设置模型输入形状
    inputs = torch.randn(1, 3, 224, 224, device='cuda')
    # 设置导出路径
    onnx_path = "models/onnx/" + model_name + ".onnx"

    model = get_model(model_name=model_name, weights_path=weights_path)

    export_onnx(model=model, onnx_path=onnx_path, inputs= inputs)
    
    # print(model)
