import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import optim
import torch
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from dataset import TaobaoDataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

# dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
dataset = TaobaoDataset("/data/ziang/data-mining/data/")
print("load done")
data = dataset[0]

_,l = data['user','buy','seller'].edge_index.shape
print(l)
#"sample 1/10"
indices = torch.randperm(l)[:int(l/10)]
# final_edge = torch.cat((data['user'].train_ind,data['user'].test_ind,indices))
final_edge = data['user'].train_ind
print(final_edge.shape)
final_edge = torch.unique(final_edge)
print(final_edge.shape)
data['user','buy','seller'].edge_index = data['user','buy','seller'].edge_index[:,final_edge]
data['user','buy','seller'].edge_attr = data['user','buy','seller'].edge_attr[final_edge]

data['seller','bought by','user'].edge_index = data['seller','bought by','user'].edge_index[:,final_edge]
data['seller','bought by','user'].edge_attr = data['seller','bought by','user'].edge_attr[final_edge]

data = T.NormalizeFeatures()(data)
data['user'].x1 = data['user'].x1.long()

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2, dropout=0):
        super().__init__()

        self.embedding_age = nn.Embedding(9,32)
        self.embedding_gender = nn.Embedding(3,32)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'buy', 'seller'): SAGEConv((-1, -1), hidden_channels),
                ('seller','bought by','user'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        self.drop_rate = dropout
        
        self.lin = Linear(hidden_channels*2, 1)

    def forward(self, x_dict, x1, edge_index_dict,mask=None):
        x_user = x1
        x_user_age = self.embedding_age(x_user[:,0])
        x_user_gender = self.embedding_gender(x_user[:,1])
        x_dict['user'] = torch.cat((x_user_age,x_user_gender,x_dict['user']),1)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            x_dict = {key: F.dropout(x.relu(), p=self.drop_rate) for key, x in x_dict.items()}
        if mask is not None:
            x_dict_user = x_dict['user'][mask[:,0].reshape(-1)]
            x_dict_seller = x_dict['seller'][mask[:,1].reshape(-1)]
            x=torch.cat((x_dict_user,x_dict_seller),1)

            x = self.lin(x)
            return torch.sigmoid(x).reshape(-1)

model = HeteroGNN(hidden_channels=64, num_layers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict,data['user'].x1, data.edge_index_dict)

def train(model, optimizer, data, train_mask,train_Y):
    # print("Training...")
    model.train()
    optimizer.zero_grad()
    # mask = data['user'].train
    out = model(data.x_dict, data['user'].x1, data.edge_index_dict,mask = train_mask)
    loss = F.mse_loss(out, train_Y)
    loss.backward()
    optimizer.step()
    # print(loss)
    return float(loss)

def evaluate(model, data, val_mask, val_Y,):

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict,data['user'].x1, data.edge_index_dict,mask = val_mask)
    # print(out.shape,val_Y.shape)
    rocauc = roc_auc_score(val_Y.cpu().numpy(),out.cpu().numpy())
    return rocauc

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
    loss = train(model, optimizer, data, train_mask,train_Y )
    rocauc = evaluate(model, data, val_mask, val_Y)
    loss_list.append(loss)
    acc_list.append(rocauc)
    # print(rocauc)

print(np.min(loss_list),np.argmin(loss_list))
print(np.max(acc_list),np.argmax(acc_list))

