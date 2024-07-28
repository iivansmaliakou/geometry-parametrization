from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

TRAIN_ITERS=1000
TRAIN_PROPORTION=0.8

# returns tensor of shape [2, 2*num_nodes-2] 
def edge_idxs(n):
    ret = []
    ret.extend([[0, n-1],[0, 1]]) # first is connected to the last
    for i in range(2, n-2):
        ret.append([i, i+2])
        ret.append([i, i-2])
    ret.extend([[n-1, n-2], [n-1, 0]]) # last is connected to the first
    return torch.tensor(ret, dtype=torch.long).t()

def make_graphs(X, conns):
    ret = []
    for i in range(X.shape[0]):
        ret.append(Data(x=torch.tensor(X[i], dtype=torch.float), edge_index=conns))
    return ret

def get_pos(torch_nodes):
    ret = {}
    for i in range(torch_nodes.shape[0]):
        ret[i] = torch_nodes[i].detach().numpy()
    return ret

class GCN(nn.Module):
    def __init__(self, num_features, connections, hidden_channels):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.connections=connections
        self.layers = nn.Sequential(
            GCNConv(num_features, hidden_channels),
            nn.Dropout(0.2),
            GCNConv(hidden_channels, num_features),
            nn.ReLU(),
        )
    
    def forward(self, X):
        X = self.layers[0](X, self.connections)
        X = self.layers[1](X)
        X = self.layers[2](X, self.connections)
        X = self.layers[3](X)
        return X

def train(model, X, loss_fn, opt, iters):
    for iter in range(iters):
        train_loader = DataLoader(X, batch_size=32, shuffle=True)
        for chunk in train_loader:
            opt.zero_grad()
            out = model(chunk.x)
            loss = loss_fn(out, chunk.x)
            loss.backward()
            opt.step()
        print(f'epoch {iter}. loss={loss.item()}')

def evaluate(model, X, conn):
    model.eval()
    rand_idx = np.random.randint(0, len(X))
    X_generated = model(X[rand_idx].x)
    fig_fantasy = plt.figure('fantasy')
    nx.draw(to_networkx(Data(X_generated, edge_index=conn)), node_size=10, pos=get_pos(X_generated))
    fig_fantasy.show()
    fig_orig = plt.figure('original')
    nx.draw(to_networkx(X[rand_idx]), node_size=10, pos=get_pos(X[rand_idx].x))
    fig_orig.show()
    plt.show()

def main():
    reg_boundary = np.load('reg_boundary.npy').reshape((500, 360, 2), order='F') # (500, 360, 2)
    train_size = int(reg_boundary.shape[0] * TRAIN_PROPORTION)
    X_train_idx = np.random.choice(reg_boundary.shape[0], train_size, replace=False)
    X_train = reg_boundary[X_train_idx]
    X_train_idx = np.sort(X_train_idx)
    X_test_idx = []
    for i in range(train_size):
        if i not in X_train_idx:
            X_test_idx.append(i)
    X_test = reg_boundary[X_test_idx]
    
    connections = edge_idxs(X_train.shape[1])
    graph_df = make_graphs(X_train, connections)
    model = GCN(X_train.shape[2], connections, 15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    train(model, graph_df, loss_fn, optimizer, TRAIN_ITERS)

    graph_df_test = make_graphs(X_test, connections)
    evaluate(model, graph_df_test, connections)
main()