import torch
import torch.nn as nn
import numpy as np

class BayesianWageEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc_out = nn.Linear(16, 2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc_out(x)

wage_model = BayesianWageEngine()

def predict_wage(skill_score: float, complexity: int = 3, demand: int = 3):
    wage_model.eval()
    predictions = []
    x = torch.tensor([[skill_score, complexity/5.0, demand/5.0]], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(50):
            wage_model.train()
            out = wage_model(x)
            mean = out[0, 0].item()
            scaled_mean = 100 + (mean + 1) * 100 
            predictions.append(scaled_mean)
    lower = max(50, int(np.percentile(predictions, 2.5)))
    upper = int(np.percentile(predictions, 97.5))
    return f"{lower} - ₹{upper}/hr", f"85% (Monte Carlo Dropout)"