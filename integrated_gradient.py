import numpy as np
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



def integrated_node_gradients(baseline, input, model, steps=50):

    # 生成步骤
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]

    # 计算梯度
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        y = model(scaled_input, data.edge_index, data.edge_attr, data.batch)[:,1]

        # 假设模型的输出是单个值
        model.zero_grad()
        y.backward()

        # 收集梯度
        grads.append(scaled_input.grad.detach().clone())

    # 计算平均梯度
    grads = torch.stack(grads).mean(dim=0)

    # 计算积分梯度
    integrated_grads = (input - baseline) * grads
    return integrated_grads

def integrated_edge_gradients(baseline, input, model, steps=50):

    # 生成步骤
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]

    # 计算梯度
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        y = model(data.x, data.edge_index, scaled_input, data.batch)[:,1]

        # 假设模型的输出是单个值
        model.zero_grad()
        y.backward()

        # 收集梯度
        grads.append(scaled_input.grad.detach().clone())

    # 计算平均梯度
    grads = torch.stack(grads).mean(dim=0)

    # 计算积分梯度
    integrated_grads = (input - baseline) * grads
    return integrated_grads


if __name__ == '__main__':
    model = Model_initial(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                          average_nodes=config['average_nodes'], \
                          out_channels=config['out_channels'], head=config['head'])
    #model = Model(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
    #              average_nodes=config['average_nodes'], \
    #              out_channels=config['out_channels'])
    checkpoint = 'save/round3/param_epoch19.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()


    node_gradients_list = []
    edge_gradients_list = []
    test_loader = load_explain(['input/data/GaoB-Q205L-GTPG-MG','input/data/GaoB-Q205L-GDPG-MG'])

    for data in tqdm(test_loader):
        input_node_features = data.x
        input_edge_features = data.edge_attr
        # 示例：创建基线和输入数据
        baseline_node_features = torch.zeros_like(input_node_features)
        baseline_edge_features = torch.zeros_like(input_edge_features)

        # 调用函数
        node_attributions = integrated_node_gradients(baseline_node_features, input_node_features, model)
        edge_attributions = integrated_edge_gradients(baseline_edge_features, input_edge_features, model)
        node_gradients_list.append(node_attributions)
        edge_gradients_list.append(edge_attributions)

        node_grads = torch.stack(node_gradients_list).mean(dim=0)
        edge_grads = torch.stack(edge_gradients_list).mean(dim=0)

        np.save('explanation/round3/node_integrated-Q205L-2type.npy', node_grads)
        np.save('explanation/round3/edge_integrated-Q205L-2type.npy', edge_grads)



