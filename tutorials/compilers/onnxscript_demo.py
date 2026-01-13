import torch
import onnxscript

# Opset 18 is the standard supported version as of PyTorch 2.6
from onnxscript import opset18 as op


# Create a model that uses the operator torch.ops.aten.add.Tensor
class Model(torch.nn.Module):
    def forward(self, input_x, input_y):
        return torch.ops.aten.add.Tensor(input_x, input_y)


# NOTE: The function signature (including parameter names) must match the signature of the unsupported PyTorch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# All attributes must be annotated with type hints.
def custom_aten_add(self, other, alpha: float = 1.0):
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    # To distinguish the custom implementation from the builtin one, we switch the order of the inputs
    return op.Add(other, self)


x = torch.tensor([1.0])
y = torch.tensor([2.0])

# Then we provide the custom implementation to the ONNX exporter as a ``custom_translation_table``.
onnx_program = torch.onnx.export(
    Model().eval(),
    (x, y),
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.add.Tensor: custom_aten_add,
    },
)
# Optimize the ONNX graph to remove redundant nodes
onnx_program.optimize()
print(onnx_program.model)

result = onnx_program(x, y)[0]
print(f"Result: {result}")
torch.testing.assert_close(result, torch.tensor([3.0]))