import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import warnings

# Suppress the minor torch.jit warning for a cleaner terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

print("🔄 Initializing GNN Fair-Wage Engine...")

# ==========================================
# 1. CREATE MOCK GRAPH DATA
# ==========================================
num_workers = 100  # Number of nodes (workers)

# Node Features: [Skill_Level (1-10), Task_Complexity (1-5), Local_Demand (1-10)]
np.random.seed(42)
skill_level = np.random.randint(1, 11, size=(num_workers, 1))
task_complexity = np.random.randint(1, 6, size=(num_workers, 1))
local_demand = np.random.randint(1, 11, size=(num_workers, 1))

# Combine into a single feature matrix (PyTorch tensor)
x = torch.tensor(np.hstack((skill_level, task_complexity, local_demand)), dtype=torch.float)

# Create random edges (connections between workers with similar skills/tasks)
edge_index = torch.randint(0, num_workers, (2, 300), dtype=torch.long)

# Target Variable: Wage (Synthetic formula: Skill*300 + Complexity*200 + Demand*150 + noise)
wage = (skill_level * 300 + task_complexity * 200 + local_demand * 150 + np.random.randint(0, 100, size=(num_workers, 1)))
y = torch.tensor(wage, dtype=torch.float)

# Create the PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
print(f"✅ Graph Created: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features per node.")

# ==========================================
# 2. DEFINE THE GCN MODEL
# ==========================================
class WageGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(WageGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize model, loss function (MSE for regression), and optimizer
model = WageGCN(num_features=data.num_node_features, hidden_dim=16)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("✅ GCN Model Defined and Ready.")

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("🔄 Training GCN for 100 epochs...")
model.train()
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1:03d} | Loss: {loss.item():.2f}")

print("✅ Training Complete!")

# ==========================================
# 4. TEST INFERENCE (Predict wage for a sample worker from our graph)
# ==========================================
model.eval() # Set to evaluation mode

# Let's pick the first worker in our graph as a sample
sample_worker_idx = 0
sample_features = data.x[sample_worker_idx]
true_wage = data.y[sample_worker_idx].item()

with torch.no_grad():
    # Pass the whole graph through to get predictions, then extract the one for our sample worker
    predictions = model(data.x, data.edge_index)
    predicted_wage = predictions[sample_worker_idx].item()
    
print("\n📝 --- WAGE PREDICTION RESULT ---")
print(f"Sample Worker Profile: Skill={int(sample_features[0].item())}, "
      f"Complexity={int(sample_features[1].item())}, "
      f"Demand={int(sample_features[2].item())}")
print(f"True Wage (Synthetic Formula) : ₹{true_wage:.2f}")
print(f"GCN Predicted Fair Wage       : ₹{predicted_wage:.2f}")
print("----------------------------------\n")
print("🎉 Day 2 GNN Module Successfully Prototyped!")