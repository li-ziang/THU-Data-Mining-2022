import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import optim
import torch
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from dataset2 import TaobaoDataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2, dropout=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'buy', 'seller'): SAGEConv((-1, -1), hidden_channels),
                ('seller','bought by','user'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        self.drop_rate = dropout
        self.lin = Linear(hidden_channels * 2, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(x.relu(), p=self.drop_rate) for key, x in x_dict.items()}
        
        if mask == None:  # used for lazy initialization
            return
        
        x_dict_user = x_dict['user'][mask[:,0].reshape(-1)]
        x_dict_seller = x_dict['seller'][mask[:,1].reshape(-1)]
        x = torch.cat((x_dict_user,x_dict_seller), 1)
        x = self.lin(x)
        return torch.sigmoid(x).reshape(-1)


def train(model, optimizer, data, train_mask, train_y):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, train_mask)
    loss = F.mse_loss(out, train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


def evaluate(model, data, val_mask, val_y):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, val_mask)
    rocauc = roc_auc_score(val_y.cpu().numpy(), out.cpu().numpy())
    return rocauc


def main():
    dataset = TaobaoDataset("data")
    print("load done")
    data = dataset[0]
    data = T.NormalizeFeatures()(data)

    model = HeteroGNN(hidden_channels=64, num_layers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, None)

    Y = data['user'].train[:,2].float()
    mask = data['user'].train
    len_y = Y.shape[0]
    indices = torch.randperm(len_y)
    train_ind = indices[:int(len_y*7/10)]
    val_ind = indices[int(len_y*7/10):]
    train_mask = mask[train_ind]
    val_mask = mask[val_ind]
    train_Y = Y[train_ind]
    val_Y = Y[val_ind]

    loss_list = []
    acc_list = []
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        loss = train(model, optimizer, data, train_mask, train_Y)
        rocauc = evaluate(model, data, val_mask, val_Y)
        loss_list.append(loss)
        acc_list.append(rocauc)
        print(rocauc)

    print(np.min(loss_list), np.argmin(loss_list))
    print(np.max(acc_list), np.argmax(acc_list))


if __name__ == '__main__':
    main()
