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

#"sample 1/10"
indices = torch.randperm(l)[:int(l/10)]
final_edge = torch.cat((data['user'].train_ind,data['user'].test_ind,indices))
data['user','buy','seller'].edge_index = data['user','buy','seller'].edge_index[:,final_edge]
data['user','buy','seller'].edge_attr = data['user','buy','seller'].edge_attr[final_edge]

data['seller','bought by','user'].edge_index = data['seller','bought by','user'].edge_index[:,final_edge]
data['seller','bought by','user'].edge_attr = data['seller','bought by','user'].edge_attr[final_edge]
data = T.NormalizeFeatures()(data)

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'buy', 'seller'): SAGEConv((-1, -1), hidden_channels),
                ('seller','bought by','user'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels*2, 1)

    def forward(self, x_dict, edge_index_dict,mask=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if mask is not None:
            x_dict_user = x_dict['user'][mask[:,0].reshape(-1)]
            print(x_dict_user.shape)
            x_dict_seller = x_dict['seller'][mask[:,1].reshape(-1)]
            x=torch.cat((x_dict_user,x_dict_seller),1)
            print(x.shape)
            x = self.lin(x)
            return torch.sigmoid(x).reshape(-1)

model = HeteroGNN(hidden_channels=64, num_layers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict, data.edge_index_dict)

def train(model, optimizer, data, train_mask,train_Y):
    print("Training...")
    model.train()
    optimizer.zero_grad()
    # mask = data['user'].train
    out = model(data.x_dict, data.edge_index_dict,mask = train_mask)
    loss = F.mse_loss(out, train_Y)
    loss.backward()
    optimizer.step()
    # print(loss)
    return float(loss)

def evaluate(model, data, val_mask, val_Y,):

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict,mask = val_mask)
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
for i in range(100):
    loss = train(model, optimizer, data, train_mask,train_Y )
    rocauc = evaluate(model, data, val_mask, val_Y)
    print(rocauc)
out = model(data.x_dict, data.edge_index_dict,mask = val_mask)
print(out)
breakpoint()
print(loss)
# def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
#         permute_masks=None, logger=None):

#     val_losses, accs, durations = [], [], []
#     for _ in range(runs):
#         data = dataset[0]
#         if permute_masks is not None:
#             data = permute_masks(data, dataset.num_classes)
#         data = data.to(device)

#         model.to(device).reset_parameters()
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         t_start = time.perf_counter()

#         best_val_loss = float('inf')
#         test_acc = 0
#         val_loss_history = []

#         for epoch in range(1, epochs + 1):
#             train(model, optimizer, data)
#             eval_info = evaluate(model, data)
#             eval_info['epoch'] = epoch

#             if logger is not None:
#                 logger(eval_info)

#             if eval_info['val_loss'] < best_val_loss:
#                 best_val_loss = eval_info['val_loss']
#                 test_acc = eval_info['test_acc']

#             val_loss_history.append(eval_info['val_loss'])
#             if early_stopping > 0 and epoch > epochs // 2:
#                 tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
#                 if eval_info['val_loss'] > tmp.mean().item():
#                     break

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         t_end = time.perf_counter()

#         val_losses.append(best_val_loss)
#         accs.append(test_acc)
#         durations.append(t_end - t_start)

#     loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

#     print(f'Val Loss: {float(loss.mean()):.4f}, '
#           f'Test Accuracy: {float(acc.mean()):.3f} ± {float(acc.std()):.3f}, '
#           f'Duration: {float(duration.mean()):.3f}')


# def train(model, optimizer, data):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()


# def evaluate(model, data):
#     model.eval()

#     with torch.no_grad():
#         logits = model(data)

#     outs = {}
#     for key in ['train', 'val', 'test']:
#         mask = data[f'{key}_mask']
#         loss = F.nll_loss(logits[mask], data.y[mask]).item()
#         pred = logits[mask].max(1)[1]
#         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

#         outs[f'{key}_loss'] = loss
#         outs[f'{key}_acc'] = acc

#     return outs

