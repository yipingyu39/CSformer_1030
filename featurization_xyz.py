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
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def cosine_similarity_matrix(matrix):
    flattened_matrix = matrix.reshape(matrix.shape[0], -1)
    num_vectors = flattened_matrix.shape[0]
    similarity_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i, num_vectors):
            similarity = cosine_similarity(flattened_matrix[i],flattened_matrix[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 对称矩阵
    return similarity_matrix

def get_features(raw_data,xyz):
    total_length = raw_data.shape[0]

    xyz_list_90ns = np.split(xyz, total_length / (90*50))
    xyz_list_90ns = xyz_list_90ns - np.mean(xyz_list_90ns, axis=0)
    xyz_list_50ns = np.split(xyz, total_length/(50*50))
    xyz_list_50ns = xyz_list_50ns - np.mean(xyz_list_50ns, axis=0)
    xyz_list_10ns = np.split(xyz, total_length/(10*50))
    xyz_list_10ns = xyz_list_10ns - np.mean(xyz_list_10ns, axis=0)
    xyz_list_5ns = np.split(xyz, total_length/(5*50))
    xyz_list_5ns = xyz_list_5ns - np.mean(xyz_list_5ns, axis=0)
    xyz_list_1ns = np.split(xyz, total_length/50)
    xyz_list_1ns = xyz_list_1ns - np.mean(xyz_list_1ns, axis=0)

    data_list_1ns = np.split(raw_data, total_length / 50)
    coord_1ns = np.array([np.mean(data,axis=0) for data in data_list_1ns])

    rp_90ns = np.array([cosine_similarity_matrix(data.transpose(1,0,2)) for data in tqdm(xyz_list_90ns)])
    rp_50ns = np.array([cosine_similarity_matrix(data.transpose(1,0,2)) for data in tqdm(xyz_list_50ns)])
    rp_10ns = np.array([cosine_similarity_matrix(data.transpose(1,0,2)) for data in tqdm(xyz_list_10ns)])
    rp_5ns = np.array([cosine_similarity_matrix(data.transpose(1,0,2)) for data in tqdm(xyz_list_5ns)])
    rp_1ns = np.array([cosine_similarity_matrix(data.transpose(1,0,2)) for data in tqdm(xyz_list_1ns)])

    print(
        'rp_90ns', rp_90ns.shape,
        'rp_50ns', rp_50ns.shape,
        'rp_10ns', rp_10ns.shape,
        'rp_5ns', rp_5ns.shape,
        'rp_1ns', rp_1ns.shape,
        'coord_1ns', coord_1ns.shape,
    )

    features_dict = {
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

    for system in systems:
        superpose_filename = os.path.join(f'../Project/GCN/input/superpose/superpose_{system}.npz')
        raw_data = load_data(superpose_filename)

        xyz_filename = os.path.join(f'../lmiformer/input/coordinate/coordinate_{system}.npz')
        xyz = load_data(xyz_filename)

        total_nums = int(raw_data.shape[0]/50)

        features_dict = get_features(raw_data,xyz)

        os.makedirs('input/data2',exist_ok=True)
        dirname = 'input/data2/'+system
        print(dirname)
        for i in tqdm(range(total_nums)):
            dict = merge_data(features_dict,i)
            os.makedirs(dirname,exist_ok=True)
            save_dict(dict,dirname,i)


