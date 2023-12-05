import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# set random seed
def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def get_ppi(ppi_file):
    ppi = np.load(ppi_file)
    all_edges = np.argwhere(ppi == 1)
    edge_data = np.empty(shape=[0, 2], dtype=int)
    edge_data = np.append(edge_data, [i for i in all_edges if i[0] != i[1]], axis=0)
    edge_data = torch.tensor(np.array(list(edge_data.T)))
    print('edges_num:', len(edge_data[0]))
    return edge_data


def get_dti(dti_file):
    cp_gene_dict = {}
    with open(dti_file, 'r') as f:
        for line in f.readlines():
            hang = line.strip().split('\t')
            genelist = hang[1].split(';')
            cp_gene_dict[hang[0]] = genelist
    return cp_gene_dict


def get_multi_hot_label(dti_dict1, dti_dict2, dti_dict3, kd_list):
    dti_dict1.update(dti_dict2)
    dti_dict1.update(dti_dict3)
    cp_label_dict, kd_label_dict = {}, {}
    for kd in kd_list:
        kd_label_dict[kd] = F.one_hot(torch.tensor(kd_list.index(kd)), num_classes=len(kd_list))
    for cp in dti_dict1:
        target_list = [n for n in dti_dict1[cp] if n in kd_list]
        if target_list == []:
            cp_label_dict[cp] = torch.tensor([0] * len(kd_list))
        else:
            index_list = [kd_list.index(n) for n in target_list]
            cp_label_dict[cp] = F.one_hot(torch.tensor(index_list), num_classes=len(kd_list)).sum(-2)
    return cp_label_dict, kd_label_dict


def get_acck_df(state, cell):
    kd_file = './dataset/kd_data/' + cell + '.l978.modz.gene.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='id')

    cp_file = './dataset/top100/' + state + '/' + cell + '.test_set.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='id')
    return kd_df, cp_df


def get_state_df(cell_lines, train_id_list, valid_id_list, test_id_list):
    all_kd_df = pd.DataFrame({})
    train_cp_df = pd.DataFrame({})
    valid_cp_df = pd.DataFrame({})
    test_cp_df = pd.DataFrame({})
    for cell in cell_lines:
        kd_file = './dataset/kd_data/' + cell + '.l978.modz.gene.tsv'
        cell_kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='id')
        print(cell, 'kd_num:', cell_kd_df.shape[1])
        all_kd_df = pd.concat([all_kd_df, cell_kd_df], axis=1)

        cp_file = './dataset/cp_data/' + cell + '.l978.modz.brd+dose+time.tsv'
        cell_cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='id')

        for brdid in train_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            train_cp_df = pd.concat([train_cp_df, brd_df], axis=1)
        for brdid in valid_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            valid_cp_df = pd.concat([valid_cp_df, brd_df], axis=1)
        for brdid in test_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            test_cp_df = pd.concat([test_cp_df, brd_df], axis=1)
        print(cell, 'train_cp_num:', train_cp_df.shape[1])
        print(cell, 'valid_cp_num:', valid_cp_df.shape[1])
        print(cell, 'test_cp_num:', test_cp_df.shape[1])
    return all_kd_df, train_cp_df, valid_cp_df, test_cp_df


def get_pairs(neg_num, cp_df, kd_df, cp_gene_dict, cell_lines_dict):
    cp_list = cp_df.columns.tolist()
    kd_list = kd_df.columns.tolist()
    exp_pair_list, pair_other_list, lable_list = [], [], []
    for cp in cp_list:
        # BRAF001_A375_24H:BRD-K63675182-003-18-6:0.15625:BRD-K63675182:0.1:24
        cell1 = cp.split(':')[0]
        brdid = cp.split(':')[1]
        dose = cp.split(':')[2]
        time = cp.split(':')[3]
        cp_exp = torch.tensor(np.array([cp_df[cp]]))
        cp_cell = torch.tensor([cell_lines_dict[cell1]])
        cp_dose = torch.tensor([float(dose)])
        cp_time = torch.tensor([float(time)])

        targets_list = [kd for kd in kd_list if kd.split(':')[1] in cp_gene_dict[brdid]]
        targets_kd_num = len(targets_list)
        if targets_kd_num == 0:
            # print('The targets of ' + brdid + ' is not in kd_df !')
            continue
        random_id_list = random.sample(list(set(kd_list).difference(set(targets_list))), neg_num * targets_kd_num)
        # KDC009_A375_96H:TRCN0000004498:-666:ZFAND6:96
        for kd in (targets_list + random_id_list):
            kd_exp = torch.tensor(np.array([kd_df[kd]]))
            kd_time = torch.tensor([float(96)])
            exp_pair = torch.cat((cp_exp, kd_exp), dim=0)
            pair_other = torch.cat((cp_cell, cp_time, cp_dose, kd_time), dim=0)
            exp_pair_list.append(exp_pair)
            pair_other_list.append(pair_other)
        lable_list += [1] * targets_kd_num + [0] * targets_kd_num * neg_num
    return torch.stack(exp_pair_list), torch.stack(pair_other_list), torch.tensor(lable_list)
