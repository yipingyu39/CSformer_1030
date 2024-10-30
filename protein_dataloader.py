from protein_dataset import MyData
from torch.utils.data import ConcatDataset,random_split
from torch_geometric.loader import DataLoader
from utils import config
from tqdm import tqdm

def load_data(folders):
    #加载数据集
    dataset = ConcatDataset([MyData(folder) for folder in tqdm(folders)])
    train_dataset,test_dataset = random_split(dataset,[int(0.8*len(dataset)),len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset,shuffle=True,batch_size=config['batch_size'])
    return train_loader, test_loader

def load_test(folder):
    #加载数据集
    dataset = MyData(folder)
    test_loader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'])
    return test_loader

def load_explain(folders):
    #加载数据集
    dataset = ConcatDataset([MyData(folder) for folder in tqdm(folders)])
    test_loader = DataLoader(dataset,shuffle=True,batch_size=1)
    return test_loader

def load_all_data(folders):
    #加载数据集
    dataset = ConcatDataset([MyData(folder) for folder in tqdm(folders)])
    data_loader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'])
    return data_loader