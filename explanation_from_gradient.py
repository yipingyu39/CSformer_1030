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
from proteinmodel import Model,Model_initial
from tqdm import tqdm

def get_gradient(model,data):

    data.x.requires_grad = True
    data.edge_attr.requires_grad = True

    out = model(data.x, data.edge_index, data.edge_attr, data.batch)

    #Loss = torch.nn.BCELoss()
    #target_output = Loss(out.to('cpu'),data.y.to('cpu'))
    target_output = out[:,1]
    print(target_output)

    target_output.backward()

    # 提取梯度
    node_gradients = data.x.grad
    edge_gradients = data.edge_attr.grad

    return node_gradients, edge_gradients


def average_gradients(gradients_list):
    # 将列表中的梯度张量堆叠成一个张量，维度为 (num_nodes, num_features)，其中 num_nodes 是节点数量，num_features 是特征维度
    stacked_gradients = torch.stack(gradients_list)

    # 对张量沿着第0维（即列表中每个节点的梯度）求和
    summed_gradients = torch.sum(stacked_gradients, dim=0)

    # 求平均
    average_gradients = summed_gradients / len(gradients_list)

    return average_gradients

def main():

    ### 加载模型
    #model = Model(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'], \
    #              out_channels=config['out_channels'])
    model = Model_initial(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'], \
                  out_channels=config['out_channels'],head=config['head'])
    checkpoint = 'save/round3/param_epoch19.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()

    ### 计算output
    node_gradients_list = []
    edge_gradients_list = []
    test_loader = load_explain(['input/data/GaoB-Q205L-GTPG-MG','input/data/GaoA-Q205L-GDPG-MG'])
    for data in tqdm(test_loader):
        node_tmp,edge_tmp = get_gradient(model,data)
        node_gradients_list.append(node_tmp)
        edge_gradients_list.append(edge_tmp)

    node_gradients = average_gradients(node_gradients_list)
    edge_gradients = average_gradients(edge_gradients_list)
    print(node_gradients.shape)
    print(edge_gradients.shape)

    np.save('explanation/round3/node_gradient-Q205L-2type.npy',node_gradients)
    np.save('explanation/round3/edge_gradient-Q205L-2type.npy',edge_gradients)

main()
