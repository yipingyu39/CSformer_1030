import numpy as np
import os
from torch_geometric.data import Data
import torch
import re
from torch.utils.data import Dataset
from utils import config
import itertools



def select_indexes(residue_number,cutoff):
    rp_raw = np.zeros([residue_number, residue_number])
    n = 0

    triu_mask = np.triu(np.ones_like(rp_raw, dtype=bool),k=1)

    filtered_indices = np.where(triu_mask)

    array = np.zeros([residue_number,residue_number])
    filtered_indices = np.where(array == 0)

    return filtered_indices


class MyData(Dataset):
    def __init__(self,dirpath):
        self.mapping = {
            'GaoA-GDPG': 0,
            #'Gao-3sn6':1,
            'GaoB-GTPG-MG':1,
            'GaoA-GTPG-MG':1,
            'GaoB-GDPG':0,
        }
        self.dirpath = dirpath
        self.filenames = [os.path.join(dirpath,filename) for filename in os.listdir(dirpath)]
        self.edge_indexes = select_indexes(330,config['cutoff'])

    def get_filenames(self):
        dirpath = self.dirpath
        filenamelist = []
        for filename in os.listdir(dirpath):
            time = re.sub('\D','',filename)
            if int(time)%450 < 360:
                filenamelist.append(os.path.join(dirpath,filename))
        return filenamelist

    def get_label(self,string):
        #label = torch.zeros(1, 2)
        #if string in self.mapping.keys():
        #    label[:,self.mapping[string]] = 1
        if string in self.mapping.keys():
            label = self.mapping[string]
        else:
            label = 0
        return label

    def get_positions(self,system):
        dirpath = f'../Project/GCN/input/coordinate/{system}'
        dirnames = os.listdir(dirpath)
        return [os.path.join(dirpath,dirname) for dirname in dirnames]

    def __getitem__(self, item):

        data = np.load(self.filenames[item],allow_pickle=True).item()
        edge_indexes = self.edge_indexes

        #residue_number = data['rp_raw'].shape[0]
        residue_number = 330
        #print('residue_number:',residue_number)
        node_indexes = range(residue_number)

        ### 保留无向边
        # 提取上三角部分（包括对角线）

        edge_features = [torch.tensor(data[key][edge_indexes]).flatten().unsqueeze(1) for key in ['rp_1ns','rp_5ns','rp_10ns','rp_50ns','rp_90ns']]
        #print('edge_features:',edge_features[0].shape)
        #node_positions = data['coord_1ns']

        #print('node_positions:',node_positions.shape)

        edge_matrix = torch.cat(edge_features,dim=1)# 330x330,7

        system = self.dirpath.split('/')[-1]
        node_positions = np.load(self.get_positions(system)[item]).astype(float)
        #print(system)
        label = self.get_label(system)
        #print('label:',label)

        ## Build Graph
        G = Data()
        G.x = torch.tensor(node_indexes, dtype=torch.float32)
        G.edge_index = torch.tensor(edge_indexes, dtype=torch.int64)
        G.edge_attr = torch.tensor(edge_matrix, dtype=torch.float32)
        G.pos  = torch.Tensor(node_positions)
        G.y = torch.tensor(label, dtype=torch.float32)
        G.x = torch.stack([G.x,G.pos],dim=1).to(torch.float32)
        return G

    def __len__(self):
        return len(self.filenames)

