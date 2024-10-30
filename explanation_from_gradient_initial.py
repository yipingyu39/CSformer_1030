import torch.nn as nn
from utils import config
import itertools
import numpy as np
import os
import torch
from torch.autograd import Variable
from protein_dataloader import load_explain
import matplotlib.pyplot as plt
import pandas as pd
from proteinmodel import Model,Model_initial,Model_for_shap
from tqdm import tqdm
from protein_dataloader import load_explain

# Function to wrap your model (if needed)
model = Model_for_shap(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'],\
                       out_channels=config['out_channels'],head=config['head'])
model.load_state_dict(torch.load('save/round3/param_epoch9.pt', map_location=torch.device('cpu')))
model = model.to(config['device'])
model.eval()

class Model_wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self,node_features,edge_features):
        batchsize = node_features.reshape(-1,330).shape[0]

        sequence = torch.tensor(range(330*batchsize), dtype=torch.float32)
        node_features = torch.stack([sequence,node_features],dim=1).to(torch.float32)

        batch = torch.tensor([ [i]* 330 for i in range(batchsize)]).reshape(-1)
        edge_index = torch.tensor([result for result in itertools.product(range(330),range(330))]).repeat(batchsize,1).T

        return self.model(node_features,edge_index,edge_features,batch)


# 假设 node_features 和 edge_features 是你的输入数据
systems = ['GaoB-GTPG-MG','GaoA-GDPG']
#systems = ['GaoA-Q205L-GDPG-MG','GaoB-Q205L-GTPG-MG']
folders = [os.path.join(config['data_dir'],system) for system in systems]
train_loader = load_explain(folders)

def get_gradient(data):
    node_features = data.x[:,1] #只要可导的部分
    edge_features = data.edge_attr
    print(node_features.shape)
    print(edge_features.shape)

    # 将输入数据转换为可求导的变量
    node_features = Variable(node_features, requires_grad=True)
    edge_features = Variable(edge_features, requires_grad=True)

    model_wrapper = Model_wrapper()
    # 前向传播
    outputs = model_wrapper(node_features, edge_features)

    # 假设我们关注的是特定的输出，比如第一个输出
    target_output = outputs[0]

    # 反向传播，计算目标输出关于输入特征的梯度
    target_output.backward()

    # 提取梯度
    node_gradients = node_features.grad
    edge_gradients = edge_features.grad
    return node_gradients,edge_gradients

    # 这里的node_gradients和edge_gradients包含了输入特征对目标输出的贡献

node_gradients = torch.zeros(21120)
edge_gradients = torch.zeros([3474240,5])
count = 0

for data in tqdm(train_loader):
    if data.x.shape[0] == node_gradients.shape[0]:
        node_tmp,edge_tmp = get_gradient(data)
        node_gradients += node_tmp
        edge_gradients += edge_tmp
        count += 1

node_gradients /= count
edge_gradients /= count

np.save('explanation/round3/node_gradient-test.npy',node_gradients)
np.save('explanation/round3/edge_gradient-test.npy',edge_gradients)
print(edge_gradients.shape)

