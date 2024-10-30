import os
import numpy as np
from tqdm import tqdm

## 1.Find files
def get_filenames(systems):
    dirname = '../Project/GCN/input/superpose'
    filenames = [os.path.join(dirname,f'superpose_{system}.npz') for system in systems]
    print('system selected:',filenames)
    return filenames

## 2.Load data
def load_data(filename):
    raw_data = np.load(filename)
    raw_data = raw_data['arr_0']
    raw_data = np.squeeze(raw_data)
    print(raw_data.shape)
    return raw_data

## 3.get_features
def get_features(raw_data):
    total_length = raw_data.shape[0]

    data_list_450ns = np.split(raw_data, total_length/22500)
    data_list_90ns = np.split(raw_data, total_length/4500)
    data_list_50ns = np.split(raw_data, total_length/2500)
    data_list_10ns = np.split(raw_data, total_length/500)
    data_list_5ns = np.split(raw_data, total_length/250)
    data_list_1ns = np.split(raw_data, total_length/50)

    coord = np.array([np.mean(data,axis=0) for data in data_list_1ns])
    coord_1ns = np.zeros_like(coord)
    coord_1ns[0, :] = coord[0, :]
    # 从第1个元素开始，每个元素减去前一个元素
    coord_1ns[1:, :] = coord[1:, :] - coord[:-1, :]

    rp_raw = np.array([np.corrcoef(raw_data.T)])
    rp_450ns = np.array([np.corrcoef(data.T) for data in data_list_450ns])
    rp_90ns = np.array([np.corrcoef(data.T) for data in data_list_90ns])
    rp_50ns = np.array([np.corrcoef(data.T) for data in data_list_50ns])
    rp_10ns = np.array([np.corrcoef(data.T) for data in data_list_10ns])
    rp_5ns = np.array([np.corrcoef(data.T) for data in data_list_5ns])
    rp_1ns = np.array([np.corrcoef(data.T) for data in data_list_1ns])

    print(
        'rp_raw:',rp_raw.shape,
        'rp_450ns', rp_450ns.shape,
        'rp_90ns', rp_90ns.shape,
        'rp_50ns', rp_50ns.shape,
        'rp_10ns', rp_10ns.shape,
        'rp_5ns', rp_5ns.shape,
        'rp_1ns', rp_1ns.shape,
        'coord_1ns', coord_1ns.shape,
    )

    features_dict = {
        'rp_raw':rp_raw,
        'rp_450ns':rp_450ns,
        'rp_90ns':rp_90ns,
        'rp_50ns':rp_50ns,
        'rp_10ns':rp_10ns,
        'rp_5ns':rp_5ns,
        'rp_1ns':rp_1ns,
        'coord_1ns':coord_1ns
    }
    return features_dict

def merge_data(features_dict,index):
    assert index >= 0 and index < 2250
    dict = {
        'rp_raw':features_dict['rp_raw'][index//2250],
        'rp_450ns':features_dict['rp_450ns'][index//450],
        'rp_90ns':features_dict['rp_90ns'][index//90],
        'rp_50ns':features_dict['rp_50ns'][index//50],
        'rp_10ns':features_dict['rp_10ns'][index//10],
        'rp_5ns':features_dict['rp_5ns'][index//5],
        'rp_1ns': features_dict['rp_1ns'][index//1],
        'coord_1ns':features_dict['coord_1ns'][index//1]
    }
    return dict

def save_dict(dict,dirname,index):
    np.save(os.path.join(dirname,'{}ns.npy'.format(index)),dict)

if __name__ == '__main__':
    systems = ['GaoA-GDPG', 'GaoB-GTPG-MG','GaoA-GTPG-MG','GaoB-GDPG','GaoA-Q205L-GTPG-MG','GaoB-Q205L-GTPG-MG','GaoA-Q205L-GDPG-MG','GaoB-Q205L-GDPG-MG']
    #systems = ['GaoA-E246K-GTPG-MG','GaoA-E246K-GDPG-MG','GaoB-E246K-GTPG-MG','GaoB-E246K-GDPG-MG','GaoA-G203R-GDPG-MG']
    filenames = get_filenames(systems)

    for filename in filenames:
        raw_data = load_data(filename)
        total_nums = int(raw_data.shape[0]/50)

        features_dict = get_features(raw_data)

        os.makedirs('input/data3',exist_ok=True)
        dirname = 'input/data3/'+filename.split('_')[1].split('.')[0]
        print(dirname)
        for i in tqdm(range(total_nums)):
            dict = merge_data(features_dict,i)
            os.makedirs(dirname,exist_ok=True)
            save_dict(dict,dirname,i)


