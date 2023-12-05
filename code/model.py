import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN_encoder(nn.Module):
    def __init__(self, drop_out):
        super(GCN_encoder, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        self.gcn = GCNConv(1, 1, cached=True, add_self_loops=False, normalize=False)
        self.fc = nn.Sequential(
            nn.Linear(978, 2048),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(2048, 100)
        )
    
    def forward(self, inputs, edges):
        inputs = inputs.reshape(-1, 1)
        output = self.gcn(inputs, edges)
        output = output.reshape(-1, 978)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc(output)
        return output
    

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5,4),
            nn.ReLU(),
            nn.Linear(4,2)
        )

    def get_r2(self, input1, input2):
        pred1 = input1 - torch.mean(input1, dim=1, keepdim=True)
        pred2 = input2 - torch.mean(input2, dim=1, keepdim=True)
        pred1_norm = torch.sqrt(torch.sum(torch.pow(pred1, 2), dim=1))
        pred2_norm = torch.sqrt(torch.sum(torch.pow(pred2, 2), dim=1))
        pred1_pred2 = torch.sum(torch.mul(pred1, pred2), dim=1)
        r2 = torch.unsqueeze(torch.pow(pred1_pred2/(pred1_norm*pred2_norm), 2), dim=1)
        return r2
        
    def forward(self, input1, input2, input_others):
        r2 = self.get_r2(input1, input2)
        output = torch.cat([r2, input_others], dim=1)
        output = self.mlp(output)
        return output


class SSGCN(nn.Module):
    def __init__(self, device, drop_out=0.3):
        super(SSGCN, self).__init__()
        self.device = device
        self.encoder = GCN_encoder(drop_out)
        self.predictor = predictor()
        
    def forward(self, input1, input2, edges, input_others):
        output1 = self.encoder(input1, edges)
        output2 = self.encoder(input2, edges)
        output = self.predictor(output1, output2, input_others)
        return output

    @torch.no_grad()
    def inference1(self, inputs, edges, chunk_size):
        num = inputs.shape[0]
        edge_num = edges.shape[1]
        chunk_edges = torch.tensor([], dtype=torch.long).to(self.device)
        for i in range(chunk_size):
            chunk_edges = torch.cat((chunk_edges, (edges+978*i)), dim=1)
        out = torch.zeros(num, 100)
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            chunk_num = min(chunk_size, num-start)
            chunk_edges = chunk_edges[:, 0: chunk_num*edge_num]
            x = inputs[start:end].to(self.device)
            x = self.encoder(x, chunk_edges)
            out[start:end] = x.cpu()
        return out

    @torch.no_grad()
    def inference2(self, inputs, chunk_size):
        num = inputs[2].shape[0]
        out = torch.zeros(num, 2)
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)
            x = self.predictor(x1, x2, z)
            out[start:end] = x.cpu()
        return out

    @torch.no_grad()
    def inference(self, inputs, edges, chunk_size):
        pair_num = inputs[2].shape[0]
        edge_num = edges.shape[1]
        chunk_edges = torch.tensor([], dtype=torch.long).to(self.device)
        for i in range(chunk_size):
            chunk_edges = torch.cat((chunk_edges, (edges+978*i)), dim=1)
        y = torch.zeros(pair_num, 2)
        for start in range(0, pair_num, chunk_size):
            end = min(start + chunk_size, pair_num)
            chunk_num = min(chunk_size, pair_num-start)
            chunk_edges = chunk_edges[:, 0: chunk_num*edge_num]
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)
            x1 = self.encoder(x1, chunk_edges)
            x2 = self.encoder(x2, chunk_edges)
            x = self.predictor(x1, x2, z)
            y[start:end] = x.cpu()
        return y