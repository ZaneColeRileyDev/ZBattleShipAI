import torch.nn as nn
import torch

class BattleShipModel(nn.Module):
    def __init__(self):
        super(BattleShipModel, self).__init__()

        self.fc1 = nn.Linear(200, 250)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(250, 200)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 100)
    
    def forward(self, x):                # x: (B, 10, 10) with values 0 / 7 / 8
        miss = (x == 7).float().flatten(1)
        hit = (x == 8).float().flatten(1)
        x = torch.cat([miss, hit], dim=1)    # (B, 200)
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        return self.fc3(x)

# weights = torch.load("checkpoints/checkpoint_9.pth", weights_only=True)
# model = BattleShipModel()
# state_dict = weights["model_state_dict"]
# model.load_state_dict(state_dict)

# torch.save(model.state_dict(), "models/BattleShipAgentV2.pth")
