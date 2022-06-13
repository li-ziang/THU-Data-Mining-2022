import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, 1))
        self.drop_rate = dropout

    def forward(self, x):
        for layer in self.layers[0:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_rate)
        x = self.layers[-1](x)
        x = torch.sigmoid(x)
        return x
