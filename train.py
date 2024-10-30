from protein_dataloader import load_all_data,load_data
from utils import config
from tqdm import tqdm
import time
import torch
import os
from sklearn.metrics import roc_auc_score
from Logger import TrainingLogger
import numpy as np
from proteinmodel import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = TrainingLogger(round='pool_ln_data_coordinate')
def Loss(type_num,label,target):
    if type_num == 2:
        loss_function = torch.nn.BCELoss()
        loss = loss_function(label,target)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        target = torch.argmax(target, axis=1)
        loss = loss_function(label,target)
    return loss


def calculate_accuracy(predictions, labels):
    # 将预测结果转换为0和1
    binary_predictions = (predictions >= 0.5).int()

    # 计算准确性
    accuracy = (binary_predictions == labels).float().mean().item()

    return accuracy

def train_model(model,train_loader,epoch,output_dir):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #激活模型
    model.train()
    label_list = []
    pred_list = []
    loss_function = torch.nn.BCELoss()

    for data in tqdm(train_loader):
        data = data.to(config['device'])
        optimizer.zero_grad()

        out = model(data.x,data.edge_index,data.edge_attr,data.batch)

        loss = loss_function(out,data.y)
        pred_list.append(out)
        label_list.append(data.y)
        logger.log(loss)

        loss.backward()
        optimizer.step()

    labels = torch.cat(label_list,dim=0)
    preds = torch.cat(pred_list,dim=0)
    logger.log("epoch {} training AUC:{:.4f}".format(epoch,roc_auc_score(labels.cpu().detach(), preds.cpu().detach())))
    os.makedirs(output_dir,exist_ok=True)
    torch.save(model.state_dict(),os.path.join(output_dir,'param_epoch{}.pt'.format(epoch)))

def test_model(model,test_loader,epoch,output_dir):
    #加载模型参数
    #model = torch.load('param_epoch{}.pt'.format(epoch))

    model.eval()
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            #print(data.y)
            data = data.to(config['device'])
            out = model(data.x,data.edge_index,data.edge_attr,data.batch)
            pred_list.append(out)
            label_list.append(data.y)
            #print(loss)

    if label_list[0].dim() > 0:
        labels = torch.cat(label_list,dim=0)
        preds = torch.cat(pred_list,dim=0)
    else:
        labels = torch.stack(label_list)
        preds = torch.cat(pred_list)
    torch.save(labels, os.path.join(output_dir,'labels.pt'))
    torch.save(preds, os.path.join(output_dir,'preds.pt'))

    #logger.log("epoch {} testing prediction accuracy:{:.4f}".format(epoch, calculate_accuracy(preds,labels)))
    logger.log("epoch {} testing AUC:{:.4f}".format(epoch, roc_auc_score(labels.cpu().detach(), preds.cpu().detach())))

def main():
    folders = config['folders']
    output_dir = config['save_path']
    os.makedirs(output_dir,exist_ok=True)

    train_loader = load_all_data(folders)
    test_loader = load_all_data(['input/data/GaoB-GDPG','input/data/GaoA-GTPG-MG'])
    print('Load Data Done')
    print(config['device'])
    #定义模型参数
    device = config['device']
    model = Model(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'],out_channels=config['out_channels'])
    #model = Model_initial(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],average_nodes=config['average_nodes'],out_channels=config['out_channels'],head=config['head'])
    #logger.log(model)
    model = Model_pool_ln(in_channels=config['in_channels'], hidden_channels=config['hidden_channels'],
                          out_channels=config['out_channels'],dropout=config['dropout'])

    logger.log(model)
    model = model.to(device)

    for e in tqdm(range(config['epoch'])):
        start = time.time()
        train_model(model,train_loader,e,output_dir=output_dir)
        test_model(model,test_loader,e,output_dir=output_dir)
        end = time.time()
        elapsed = end-start
        print(f'{elapsed:.2f} seconds')

main()