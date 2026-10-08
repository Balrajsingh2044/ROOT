import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("🔄 Initializing Bayesian GNN Fair-Wage Engine...")

# ==========================================
# 1. REUSE MOCK GRAPH DATA (From Day 2)
# ==========================================
num_workers = 100
np.random.seed(42)
skill_level = np.random.randint(1, 11, size=(num_workers, 1))
task_complexity = np.random.randint(1, 6, size=(num_workers, 1))
local_demand = np.random.randint(1, 11, size=(num_workers, 1))

x = torch.tensor(np.hstack((skill_level, task_complexity, local_demand)), dtype=torch.float)
edge_index = torch.randint(0, num_workers, (2, 300), dtype=torch.long)
wage = (skill_level * 300 + task_complexity * 200 + local_demand * 150 + np.random.randint(0, 100, size=(num_workers, 1)))
y = torch.tensor(wage, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=y)

# ==========================================
# 2. DEFINE THE GCN MODEL (With Dropout)
# ==========================================
class BayesianWageGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(BayesianWageGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout remains active during inference for Monte Carlo estimation
        x = F.dropout(x, p=0.3, training=True) # Note: training=True forces dropout on!
        x = self.conv2(x, edge_index)
        return x

model = BayesianWageGCN(num_features=data.num_node_features, hidden_dim=16)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Quick training to get baseline weights
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model(data.x, data.edge_index), data.y)
    loss.backward()
    optimizer.step()

print("✅ Baseline Model Trained. Switching to Bayesian Inference...")

# ==========================================
# 3. MONTE CARLO DROPOUT INFERENCE
# ==========================================
# We do NOT call model.eval(). We keep it in train mode to keep Dropout active,
# but we wrap it in torch.no_grad() so we don't update weights.
model.train() 

sample_worker_idx = 0
sample_features = data.x[sample_worker_idx]
num_samples = 50 # Number of Monte Carlo passes

predictions = []
with torch.no_grad():
    for _ in range(num_samples):
        pred = model(data.x, data.edge_index)
        predictions.append(pred[sample_worker_idx].item())

# Calculate Mean (Expected Wage) and Standard Deviation (Uncertainty)
mean_wage = np.mean(predictions)
std_wage = np.std(predictions)

# 95% Confidence Interval is roughly Mean ± (1.96 * Std Dev)
confidence_interval = 1.96 * std_wage
lower_bound = mean_wage - confidence_interval
upper_bound = mean_wage + confidence_interval

print("\n📝 --- BAYESIAN WAGE PREDICTION RESULT ---")
print(f"Sample Worker Profile: Skill={int(sample_features[0].item())}, "
      f"Complexity={int(sample_features[1].item())}, "
      f"Demand={int(sample_features[2].item())}")
print(f"Expected Fair Wage      : ₹{mean_wage:.2f}")
print(f"Uncertainty (Std Dev)   : ± ₹{std_wage:.2f}")
print(f"95% Confidence Interval : [₹{lower_bound:.2f} , ₹{upper_bound:.2f}]")
print("---------------------------------------------")
print("💡 Presentation Note: This range protects both the worker and the platform")
print("   by acknowledging data uncertainty, preventing rigid, unfair lowball offers.\n")