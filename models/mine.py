import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2

    def mi(self, x, y):
        y_shuffle = y[torch.randperm(y.size(0))].cuda()
        pred_xy = self.forward(x, y)
        pred_x_y = self.forward(x, y_shuffle)
        mine = - torch.mean(pred_xy) + torch.log(torch.mean(torch.exp(pred_x_y)))
        return mine

