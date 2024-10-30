import torch

config = {
    'batch_size':64,
    'device':torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'folders':['input/data/GaoB-GTPG-MG','input/data/GaoA-GDPG'],
    "in_channels":2,
    "hidden_channels":16,
    "average_nodes":330,
    "out_channels":1,#3,
    'logger':'save/pool_ln_data_coordinate/model.log',
    'save_path':'save/pool_ln_data_coordinate',
    'head':1,
    'epoch':10,
    'cutoff':0,
    'data_dir':'input/data/',
    'dropout':0.0,
}