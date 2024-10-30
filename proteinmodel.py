from torch.utils.data import Dataset,ConcatDataset,random_split
import os
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv,DenseGraphConv, dense_mincut_pool, TopKPooling
from torch_geometric.nn import ResGatedGraphConv,EdgeConv,GATConv,TransformerConv,GATv2Conv
from torch_geometric.utils import to_dense_adj,to_dense_batch
import itertools
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from math import ceil
import time

## Configuration
from utils import config

## Model

class Model_initial(nn.Module):
    def __init__(self,in_channels, hidden_channels,average_nodes,out_channels,head):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        num_nodes = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels*head, num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels*head, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)
        #self.initialize_weights()


    def forward(self,x,edge_index,edge_attr,batch):
        # 加载数据
        #x = self.pooling(x=x,edge_index=edge_index,edge_attr=edge_attr)
        #x= torch.cat([x.unsqueeze(1),position],dim=1)
        # 线性层
        x = self.norm(x)
        x = self.conv1(x, edge_index,edge_attr).relu()

        x, mask = to_dense_batch(x, batch)
        #adj = to_dense_adj(edge_index,batch,edge_attr)
        #adj = adj.mean(dim=-1)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)

        x = x.mean(dim=1)

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

    def initialize_weights(self):
        m = self.modules()
        if isinstance(m, nn.Linear):
            # You can adjust the gain for the Xavier initialization to see if it helps
            nn.init.orthogonal(m.weight)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)

class Model_for_shap(nn.Module):
    def __init__(self,in_channels, hidden_channels,average_nodes,out_channels,head):
        #初始化
        super().__init__()

        #self.pooling = TopKPooling(in_channels=1)
        self.norm = nn.BatchNorm1d(in_channels)
        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        num_nodes = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels*head, num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels*head, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)
        #self.initialize_weights()


    def forward(self,x,edge_index,edge_attr,batch):
        # 加载数据
        #x = self.pooling(x=x,edge_index=edge_index,edge_attr=edge_attr)
        #x= torch.cat([x.unsqueeze(1),position],dim=1)
        # 线性层
        x = self.norm(x)
        x = self.conv1(x, edge_index,edge_attr).relu()

        x, mask = to_dense_batch(x, batch)
        #adj = to_dense_adj(edge_index,batch,edge_attr)
        #adj = adj.mean(dim=-1)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1

        return class_1

    def initialize_weights(self):
        m = self.modules()
        if isinstance(m, nn.Linear):
            # You can adjust the gain for the Xavier initialization to see if it helps
            nn.init.orthogonal(m.weight)

class Model(nn.Module):
    def __init__(self,in_channels, hidden_channels,average_nodes,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.norm2 = nn.BatchNorm1d(num_nodes_1)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.norm3 = nn.BatchNorm1d(num_nodes_2)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)
        #self.initialize_weights()


    def forward(self,x,edge_index,edge_attr,batch):
        adj = to_dense_adj(edge_index,batch)

        x = self.conv1(x, edge_index,edge_attr)
        x = self.norm1(x).relu()
        x, mask = to_dense_batch(x, batch)
        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s)

        x = self.conv2(x, adj)
        x = self.norm2(x).relu()
        s = self.mlp2(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)
        x = self.norm3(x).relu()
        s = self.mlp3(x)
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = x.mean(dim=1)

        ###
        #输出部分
        x = self.lin1(x).relu()
        output = self.readout1(x).sigmoid()
        #print(output)

        return output.squeeze(-1)

    def initialize_weights(self):
        m = self.modules()
        if isinstance(m, nn.Linear):
            # You can adjust the gain for the Xavier initialization to see if it helps
            nn.init.orthogonal(m.weight)

class Model_simple(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)
        #self.initialize_weights()


    def forward(self,x,edge_index,edge_attr,batch):
        # 加载数据
        #x = x.view(config['batch_size'], 330, config['in_channels'])
        #edge_attr = edge_attr.view(config['batch_size'], 54285, 5)
        # 线性层\
        #x[:, 1] = x[:, 0]
        x = self.norm(x)
        x = self.conv1(x, edge_index,edge_attr)
        x = self.bn1(x).relu()

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = x.permute(0, 2, 1)  # 将形状从 [batch_size, 21120, 16] 转换为 [batch_size, 16, 21120]
        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1)  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_complex(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)
        #self.initialize_weights()


    def forward(self,x,edge_index,edge_attr,batch):
        # 加载数据
        #x = x.view(config['batch_size'], 330, config['in_channels'])
        #edge_attr = edge_attr.view(config['batch_size'], 54285, 5)
        # 线性层\
        x[:, 1] = x[:, 0]
        x = self.conv1(x, edge_index,edge_attr)
        x = self.bn1(x).relu()

        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()

        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = x.permute(0, 2, 1)  # 将形状从 [batch_size, 21120, 16] 转换为 [batch_size, 16, 21120]
        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1)  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_base(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)

        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):

        x = self.conv1(x, edge_index,edge_attr).relu()

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = x.permute(0, 2, 1).contiguous()  # 将形状从 [batch_size, 21120, 16] 转换为 [batch_size, 16, 21120]
        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1).contiguous()  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_bn(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):

        x = self.conv1(x, edge_index,edge_attr)

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = x.permute(0, 2, 1)  # 将形状从 [batch_size, 330, 16] 转换为 [batch_size, 16, 330]

        x = self.bn1(x).relu()
        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1).contiguous()  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):

        x = self.conv1(x, edge_index,edge_attr).relu()

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_ln(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.ln1 = torch.nn.LayerNorm(330)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):

        x = self.conv1(x, edge_index,edge_attr)

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = self.ln1(x).relu()

        x = x.permute(0, 2, 1).contiguous()  # 将形状从 [batch_size, 330, 16] 转换为 [batch_size, 16, 330]

        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1).contiguous()  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool_bn(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels,dropout=None):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        if dropout == None:
            self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        else:
            self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=5,dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):
        batch_size = int(batch.shape[0]/330)

        x = self.conv1(x, edge_index,edge_attr)
        node_num = int(x.shape[0] / batch_size)
        x = x.reshape(batch_size, node_num, config['hidden_channels']).permute(0, 2, 1).contiguous()
        x = self.bn1(x).relu().permute(0, 2, 1).contiguous()
        x = x.reshape(batch_size*node_num, config['hidden_channels'])

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj) #[64,165,16]
        x = x.permute(0, 2, 1).contiguous() #[64,16,165]
        x = self.bn2(x).relu().permute(0, 2, 1).contiguous() #[64,165,16]
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        x = x.permute(0, 2, 1)
        x = self.bn3(x).relu().permute(0, 2, 1).contiguous()

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool_ln(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels,dropout):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5,dropout=dropout)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln3 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):
        batch_size = int(batch.shape[0]/330)

        x = self.conv1(x, edge_index,edge_attr) #[21120,16]
        node_num = int(x.shape[0] / batch_size)
        x = x.reshape(batch_size, node_num, config['hidden_channels']) #[64,330,16]
        x = self.ln1(x).relu()
        x = x.reshape(batch_size*node_num, config['hidden_channels'])

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj) #[64,165,16]
        x = self.ln2(x).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        x = self.ln3(x).relu()

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool_norm(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln3 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):
        batch_size = int(batch.shape[0]/330)

        x = self.conv1(x, edge_index,edge_attr) #[21120,16]
        x = self.bn1(x).relu()

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj) #[64,165,16]
        x = self.ln2(x).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        x = self.ln3(x).relu()

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool_ln_dropout(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels,dropout):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5,dropout=dropout)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln3 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):
        batch_size = int(batch.shape[0]/330)

        x = self.conv1(x, edge_index,edge_attr) #[21120,16]
        node_num = int(x.shape[0] / batch_size)
        x = x.reshape(batch_size, node_num, config['hidden_channels']) #[64,330,16]
        x = self.ln1(x).relu()
        x = x.reshape(batch_size*node_num, config['hidden_channels'])

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj) #[64,165,16]
        x = self.ln2(x).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        x = self.ln3(x).relu()

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_pool_ln_complex(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.ln1 = torch.nn.LayerNorm(hidden_channels)

        average_nodes = config['average_nodes']
        num_nodes_1 = ceil(0.5 * average_nodes)
        self.mlp1 = nn.Linear(hidden_channels, num_nodes_1)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln2 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_2 = ceil(0.5 * num_nodes_1)
        self.mlp2 = nn.Linear(hidden_channels, num_nodes_2)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.ln3 = torch.nn.LayerNorm(hidden_channels)
        num_nodes_3 = ceil(0.5 * num_nodes_2)
        self.mlp3 = nn.Linear(hidden_channels, num_nodes_3)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):
        batch_size = int(batch.shape[0]/330)

        x = self.conv1(x, edge_index,edge_attr) #[21120,16]
        node_num = int(x.shape[0] / batch_size)
        x = x.reshape(batch_size, node_num, config['hidden_channels']) #[64,330,16]
        x = self.ln1(x).relu()
        x = x.reshape(batch_size*node_num, config['hidden_channels'])

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index,batch)

        s = self.mlp1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj) #[64,165,16]
        x = self.ln2(x).relu()
        s = self.mlp2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)
        x = self.conv3(x, adj)
        x = self.ln3(x).relu()

        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)

class Model_bn(nn.Module):
    def __init__(self,in_channels, hidden_channels,out_channels):
        #初始化
        super().__init__()
        #self.pooling = TopKPooling(in_channels=1)
        #self.norm = nn.BatchNorm1d(in_channels)

        self.conv1 = TransformerConv(in_channels, hidden_channels,edge_dim=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.readout1 = nn.Linear(hidden_channels, out_channels)


    def forward(self,x,edge_index,edge_attr,batch):

        x = self.conv1(x, edge_index,edge_attr)

        batch_size = int(batch.shape[0]/330)
        x = x.view(batch_size, 330, config['hidden_channels'])
        x = x.permute(0, 2, 1)  # 将形状从 [batch_size, 330, 16] 转换为 [batch_size, 16, 330]

        x = self.bn1(x).relu()
        x = self.pool1(x)  # 应用第一个最大池化
        x = self.pool2(x)  # 应用第二个最大池化
        x = self.pool3(x)  # 应用第三个最大池化
        x = x.permute(0, 2, 1).contiguous()  # 将形状转换回 [batch_size, 64, 16]
        ###
        #输出部分
        x = self.lin1(x).relu()
        class_1 = self.readout1(x).mean(axis=1)
        class_1 = class_1.sigmoid()
        #class_1 = F.softmax(self.readout1(x))
        #return class_1.squeeze() if class_1.dim() > 1 else class_1
        return class_1.squeeze(-1)