import torch
import onnxscript

# Opset 18 is the standard supported version as of PyTorch 2.6
from onnxscript import opset18 as op
class GeluModel(torch.nn.Module):
    def forward(self, input_x):
        return torch.ops.aten.gelu(input_x)


# Create a namespace for the custom operator using ONNX Script
# ``com.microsoft`` is an official ONNX Runtime namespace
microsoft_op = onnxscript.values.Opset(domain="com.microsoft", version=1)

# NOTE: The function signature (including parameter names) must match the signature of the unsupported PyTorch operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
# The function must be scripted using the ``@onnxscript.script()`` decorator when
# using operators from custom domains. This may be improved in future versions.
from onnxscript import FLOAT


@onnxscript.script(microsoft_op)
def custom_aten_gelu(self: FLOAT, approximate: str = "none") -> FLOAT:
    return microsoft_op.Gelu(self)
x = torch.tensor([1.0])

onnx_program = torch.onnx.export(
    GeluModel().eval(),
    (x,),
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.gelu.default: custom_aten_gelu,
    },
)

# Optimize the ONNX graph to remove redundant nodes
onnx_program.optimize()
print(onnx_program.model)

result = onnx_program(x)[0]
print(f"Result: {result}")
torch.testing.assert_close(result, torch.ops.aten.gelu(x))