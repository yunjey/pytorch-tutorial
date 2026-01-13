import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import onnxscript
import os


def get_version():
    print(torch.__version__)
    print(onnxscript.__version__)
    print(onnxruntime.__version__)


"""简单的图像分类器模型"""


class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def onnx_demo():
    # 创建示例输入（固定随机种子以确保可重复性）
    torch.manual_seed(42)
    example_inputs = (torch.randn(1, 1, 32, 32),)
    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
    print(f"Input length: {len(onnx_inputs)}")
    print(f"Sample input shape: {onnx_inputs[0].shape}")

    # 创建一个PyTorch模型实例，用于导出和比较
    torch_model = ImageClassifierModel()

    # 生成ONNX模型（传入同一个模型实例）
    model_2_onnx(torch_model, example_inputs)

    # 加载ONNX模型并运行推理
    ort_session = onnxruntime.InferenceSession(
        "./image_classifier_model.onnx", providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in
                         zip(ort_session.get_inputs(), onnx_inputs)}

    # ONNX Runtime returns a list of outputs
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    # 使用同一个PyTorch模型进行推理
    torch_outputs = torch_model(*example_inputs)

    print(f"PyTorch output shape: {torch_outputs.shape}")
    print(f"ONNX Runtime output shape: {onnxruntime_outputs.shape}")

    # 直接比较整个张量而不是逐个元素比较
    try:
        torch.testing.assert_close(torch_outputs, torch.tensor(onnxruntime_outputs), rtol=1e-3, atol=1e-3)
        print("PyTorch and ONNX Runtime output matched!")
    except AssertionError as e:
        print(f"Outputs didn't match: {e}")
        # 输出详细的差异信息
        print("\nPyTorch output:")
        print(torch_outputs)
        print("\nONNX Runtime output:")
        print(onnxruntime_outputs)
        print("\nDifference:")
        print(torch_outputs - torch.tensor(onnxruntime_outputs))

    print(f"Output length: {onnxruntime_outputs.shape[1]}")
    print(f"Sample output: {onnxruntime_outputs[0][:5]}...")


def model_2_onnx(torch_model, example_inputs):
    # 导出模型为ONNX格式，使用默认的opset版本
    torch.onnx.export(
        torch_model,
        example_inputs,
        "image_classifier_model.onnx",
        export_params=True,
        # 不指定opset_version，让PyTorch自动选择合适的版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # 验证导出的模型
    onnx_model = onnx.load("image_classifier_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model generated and validated successfully!")
    print(f"ONNX model opset version: {onnx_model.opset_import[0].version}")


if __name__ == '__main__':
    onnx_demo()