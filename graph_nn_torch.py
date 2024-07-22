import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np

X = torch.from_numpy(np.load('reg_boundary.npy')).to(torch.float32)

def create_edge_index(num_points):
    edge_index = []
    for i in range(num_points - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    return torch.tensor(edge_index, dtype=torch.long).t()

edge_index = create_edge_index(X.shape[1])

def create_data_objects(X, edge_index):
    data_list = []
    for i in range(X.shape[0]):
        data = Data(x=X[i], edge_index=edge_index)
        data_list.append(data)
    return data_list

data_list = create_data_objects(X, edge_index)

train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNNModel(in_channels=2, hidden_channels=64, out_channels=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')

for epoch in range(50):
    print(f'Epoch {epoch+1}')
    train()

model.eval()
with torch.no_grad():
    sample_data = data_list[5]
    generated_points = model(sample_data)
    np.save('generated_data/test_sample_graph.npy', sample_data.x.numpy())
    np.save('generated_data/pred_graph.npy', generated_points.numpy())