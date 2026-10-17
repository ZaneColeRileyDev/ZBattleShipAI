import torch
import numpy as np
import onnx

from model import BattleShipModel

weights = torch.load("models/BattleShipAgentV2.pth", map_location="cpu", weights_only=True)

model = BattleShipModel()
model.load_state_dict(weights)

dummy_input = torch.zeros((1, 10, 10), dtype=torch.float32)


torch.onnx.export(
    model,
    dummy_input,
    "models/BattleShipAgent.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

model_onnx = onnx.load("models/BattleShipAgent.onnx")
onnx.checker.check_model(model_onnx)
print("Export complete and verified successfully!")

