import os
import sys 
import random
import time
import pickle as pkl
import warnings
import copy

import utils
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn import metrics


def train(dataloader):
    net.train()
    for batch, data in enumerate(dataloader, 0):
        x, y, z = data
        bs = len(z)
        batch_edges = torch.tensor([], dtype=torch.long).to(device)
        for i in range(bs):
            batch_edges = torch.cat((batch_edges, (edge_data + 978 * i)), dim=1)
        batch_input1 = x[:, 0].to(device)
        batch_input2 = x[:, 1].to(device)
        batch_other = y.to(device)
        batch_label = z.to(device)

        output = net(batch_input1, batch_input2, batch_edges, batch_other)
        loss = criterion(output, batch_label.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(dataloader):
    net.eval()
    batch_loss, pairs_num = 0.0, 0
    label_list, class_list, score_list = [], [], []
    for batch, data in enumerate(dataloader, 0):
        x, y, z = data
        bs = len(z)
        batch_edges = torch.tensor([], dtype=torch.long).to(device)
        for i in range(bs):
            batch_edges = torch.cat((batch_edges, (edge_data + 978 * i)), dim=1)

        batch_input1 = x[:, 0].to(device)
        batch_input2 = x[:, 1].to(device)
        batch_other = y.to(device)
        batch_label = z.to(device)
        label_list += list(batch_label.cpu().numpy())

        output = net(batch_input1, batch_input2, batch_edges, batch_other)
        loss = criterion(output, batch_label.long())
        batch_loss += loss.item() * bs
        pairs_num += bs

        out_score = nn.Softmax(dim=1)(output)
        out_class = torch.argmax(out_score, 1)
        cpi_score = out_score[:, -1]
        class_list += list(out_class.cpu().numpy())
        score_list += list(cpi_score.cpu().numpy())
    indicator_list = [label_list, class_list, score_list]
    epoch_loss = batch_loss / pairs_num
    torch.cuda.empty_cache()
    return epoch_loss, indicator_list


@torch.no_grad()
def topk_acc(cp_df, kd_df, cell, label_dict1, label_dict2):
    net.eval()
    cp_num = cp_df.shape[1]
    kd_num = kd_df.shape[1]
    
    brd_list = [cp.split(':')[1] for cp in cp_df for j in range(kd_num)]
    gene_list = [kd.split(':')[1] for kd in kd_df] * cp_num
    
    cell_others = torch.tensor([[cell_lines_dict[cell]]]*cp_num*kd_num)
    others = torch.tensor([[24,10,96]]*cp_num*kd_num)
    others = torch.cat((cell_others, others),dim=1)

    cp_data = torch.tensor(cp_df.values).T.to(device)
    kd_data = torch.tensor(kd_df.values).T.to(device)
    feature1 = net.inference1(cp_data.float(), edge_data, 256)
    feature2 = net.inference1(kd_data.float(), edge_data, 256)
    
    feature1 = feature1.repeat_interleave(kd_num, 0)
    feature2 = feature2.repeat(cp_num, 1)
    inputs = [feature1.float(), feature2.float(), others.float()]
    output = net.inference2(inputs, 2048)
    
    out_score = F.softmax(output, dim=1)
    cpi_class = list(torch.argmax(out_score,1).numpy())
    cpi_score = list(out_score[:,-1].numpy())
    score_matrix = (out_score[:,-1].view(cp_num, kd_num))    
    
    cp_label = torch.stack([label_dict1[cp.split(':')[1]] for cp in cp_df])
    kd_label = torch.stack([label_dict2[kd.split(':')[1]] for kd in kd_df])
    mask = torch.matmul(cp_label, kd_label.T)
    target = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
    cpi_label = list(target.view(-1).numpy())
    
    b, q = target.size()
    pred = torch.argsort(score_matrix, 1, descending = True)
    pred = F.one_hot(pred[:, 0:top_k], num_classes = q).sum(-2)
    pred = (pred * target).sum(-1)
    preds = torch.where(pred > 0, torch.ones_like(pred), torch.zeros_like(pred))
    acck = torch.sum(preds, dtype = float).item() * 100.0 / b
    
    lists = [cpi_label, cpi_class, cpi_score, brd_list, gene_list]
    return acck, lists


def loader(all_cp_df):
    exps, labels, others = utils.get_pairs(neg_num, all_cp_df, all_kd_df, dti_dict1, cell_lines_dict)
    print(exps.size(), others.size(), labels.size())
    print('exps memory: {:04f} Gb'.format(exps.element_size() * exps.nelement() / 1024 / 1024 / 1024))
    data_set = TensorDataset(exps.float(), labels, others.float())
    data_loader = DataLoader(dataset=data_set, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
    return data_loader


def indicator(indicator_list):
    label_list, class_list, score_list = indicator_list
    acc = metrics.accuracy_score(label_list, class_list)
    precision = metrics.precision_score(label_list, class_list)
    recall = metrics.recall_score(label_list, class_list)
    f1 = metrics.f1_score(label_list, class_list)
    AUROC = metrics.roc_auc_score(label_list, score_list)
    AUPRC = metrics.average_precision_score(label_list, score_list)
    indicator_dict = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'AUROC': AUROC, 'AUPRC': AUPRC}
    return acc, precision, recall, f1, AUROC, AUPRC


def weight_initialize(net):
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')

config = {'seed': 888, 'neg_num': 3, 'batch_size': 256, 
          'drop_out': 0.3, 'learning_rate': 0.001, 'weight_decay': 0.001}
print(config)
seed = config['seed']
utils.seed_torch(seed)
utils.setup_cpu(1)
top_k = 100
neg_num = config['neg_num']
batch_size = config['batch_size']
drop_out = config['drop_out']
lr = config['learning_rate']
weight_decay = config['weight_decay']
warnings.filterwarnings('ignore')
# all_available_cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']
cell_lines = ['PC3']
cell_lines_dict = {'A375': 8, 'A549': 7, 'HA1E': 6, 'HCC515': 5, 'HT29': 4, 'MCF7': 3, 'PC3': 2, 'VCAP': 1}


ppi_file = './dataset/ppi_expression.npy'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
edge_data = utils.get_ppi(ppi_file).to(device)
net = model.SSGCN(device).to(device)
weight_initialize(net)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

kd_gene_file = './dataset/kd_genes.txt'
train_id_file = './dataset/train_id.txt'
valid_id_file = './dataset/valid_id.txt'
test_id_file = './dataset/test_id.txt'
kd_gene_list = pd.read_csv(kd_gene_file, sep='\t', header=None)[0].tolist()
train_id_list = pd.read_csv(train_id_file, sep='\t', header=None)[0].tolist()
valid_id_list = pd.read_csv(valid_id_file, sep='\t', header=None)[0].tolist()
test_id_list = pd.read_csv(test_id_file, sep='\t', header=None)[0].tolist()

dti_file1 = './dataset/phase1.dti.txt'
dti_file2 = './dataset/phase2.dti.txt'
dti_file3 = './dataset/pabons.dti.txt'
dti_dict1 = utils.get_dti(dti_file1)
dti_dict2 = utils.get_dti(dti_file2)
dti_dict3 = utils.get_dti(dti_file3)
kd_gene_file = './dataset/kd_genes.txt'
kd_gene_list = pd.read_csv(kd_gene_file, sep='\t', header=None)[0].tolist()
cp_label_dict, kd_label_dict = utils.get_multi_hot_label(dti_dict1, dti_dict2, dti_dict3, kd_gene_list)

all_kd_df, train_cp_df, valid_cp_df, test_cp_df = utils.get_state_df(cell_lines, train_id_list, valid_id_list,
                                                                        test_id_list)

##data loading
print('data loading...................')
start_time = time.time()
train_dataloader = loader(train_cp_df)
valid_dataloader = loader(valid_cp_df)
test_dataloader = loader(test_cp_df)

with open('./pickle/train_dataloader.pickle', 'wb') as train_data_file:
    pkl.dump(train_dataloader, train_data_file)
with open('./pickle/valid_dataloader.pickle', 'wb') as valid_data_file:
    pkl.dump(valid_dataloader, valid_data_file)
with open('./pickle/test_dataloader.pickle', 'wb') as test_data_file:
    pkl.dump(test_dataloader, test_data_file)

end_time = time.time()
print('time for data loading: {:.5f}'.format(end_time - start_time))

train_loss_history, valid_loss_history, test_loss_history = [], [], []
train_F1_history, valid_F1_history, test_F1_history = [], [], []
train_AUROC_history, valid_AUROC_history, test_AUROC_history =[], [], []
train_AUPRC_history, valid_AUPRC_history, test_AUPRC_history = [], [], []
all_time, best_AUPRC = 0.0, 0.0
for epoch in range(1, 1000):
    start_time = time.time()
    print('#########', '\n', 'Epoch {:04d}'.format(epoch))
    train(train_dataloader)

    loss, indicator_list = evaluate(train_dataloader)
    train_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    train_F1_history.append(f1)
    train_AUROC_history.append(AUROC)
    train_AUPRC_history.append(AUPRC)
    print(
        'Tra: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))

    loss, indicator_list = evaluate(valid_dataloader)
    valid_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    valid_F1_history.append(f1)
    valid_AUROC_history.append(AUROC)
    valid_AUPRC_history.append(AUPRC)
    print(
        'Val: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))

    # copy best valid_acck model
    if AUPRC > best_AUPRC:
        best_epoch = epoch
        best_AUPRC = AUPRC
        best_model = copy.deepcopy(net)
        best_optimizer = copy.deepcopy(optimizer)

    loss, indicator_list = evaluate(test_dataloader)
    test_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    test_F1_history.append(f1)
    test_AUROC_history.append(AUROC)
    test_AUPRC_history.append(AUPRC)
    print(
        'Tes: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))

    # Early stop
    if epoch >= 300 and valid_AUPRC_history[-1] <= np.mean(valid_AUPRC_history[-21:-1]):
        print('#########', '\n', 'Early stopping...')
        break

    end_time = time.time()
    all_time += end_time - start_time
    print('epoch_time = {:.5f}'.format(end_time - start_time))
print('Optimization Finished! Stop at epoch:', epoch, 'time= {:.5f}'.format(all_time))

print('###save_model###')
state = {'net': best_model.state_dict(), 'optimizer': best_optimizer.state_dict(), 'epoch': best_epoch}
file_name = 'best_epoch_{}_bs_{}_lr_{}_wd_{}.ssgcn.pth'.format(best_epoch, batch_size, lr, weight_decay)
torch.save(state, f'./saved_model/{file_name}')
print(file_name)


net = model.SSGCN(device).to(device)
file = f'./saved_model/{file_name}'
net.load_state_dict(torch.load(file)['net'],strict=False)


print('#########Topk acc test in phase1')
for cell in ['A375','A549','HA1E','HCC515','HT29','MCF7','PC3','VCAP']:
    cell_kd_df, cell_cp_df = utils.get_acck_df('phase1', cell)
    test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict)
    print(cell, test_acck1)

print('#########Topk acc test in pabon')
for cell in ['A375','A549','HA1E','HCC515','HT29','MCF7','PC3','VCAP']:
    cell_kd_df, cell_cp_df = utils.get_acck_df('pabons', cell)
    test_acck2, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict)
    print(cell, test_acck2)