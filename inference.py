from protein_dataloader import load_test
from utils import config
from tqdm import tqdm
import time
import torch
import os
from sklearn.metrics import roc_auc_score
from Logger import TrainingLogger
from proteinmodel import Model,Model_initial,Model_simple

def test_model(model,test_loader,output_dir):

    model.eval()
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(torch.device('cpu'))
            out = model(data.x,data.edge_index,data.edge_attr,data.batch)
            pred_list.append(out)
            label_list.append(data.y)

    if label_list[0].dim() > 0:
        preds = torch.cat(pred_list,dim=0)
    else:
        preds = torch.cat(pred_list)
    torch.save(preds, os.path.join(output_dir,'preds-round12-epoch9.pt'))

def main():

    systems = ['GaoA-GDPG','GaoB-GTPG-MG','GaoA-GTPG-MG','GaoB-GDPG','GaoA-Q205L-GTPG-MG','GaoB-Q205L-GTPG-MG','GaoA-Q205L-GDPG-MG','GaoB-Q205L-GDPG-MG']
    #systems = ['GaoA-GDPG','GaoB-GTPG-MG','Gao-3sn6']
    checkpoint = 'save/round12/param_epoch9.pt'
    #加载模型参数
    #model = Model(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'],out_channels=config['out_channels'])
    model = Model_initial(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                          average_nodes=config['average_nodes'], out_channels=config['out_channels'],
                          head=config['head'])
    #model = Model_simple(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],out_channels=config['out_channels'])
    model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')))

    for system in tqdm(systems):
        test_loader = load_test(f'input/data/{system}')
        print('Load Data Done')
        output_dir = f'prediction/{system}'
        os.makedirs(output_dir,exist_ok=True)
        test_model(model,test_loader,output_dir=output_dir)


main()