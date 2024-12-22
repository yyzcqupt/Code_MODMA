import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from scipy.signal import stft

def extract_frequency_bands(eeg_data, fs, window='hann', nperseg=256, noverlap=None):
    # Define frequency bands
    bands = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 50)]
    num_channels, num_timepoints = eeg_data.shape
    num_bands = len(bands)

    # Calculate STFT
    f, t, Zxx = stft(eeg_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, axis=-1)

    # Initialize the output array
    T_2 = len(t)
    band_features = np.zeros((num_channels, T_2, num_bands))

    # Extract features for each band
    for idx, (low, high) in enumerate(bands):
        band_mask = (f >= low) & (f < high)
        band_power = np.abs(Zxx[:, band_mask, :])**2
        band_features[:, :, idx] = band_power.mean(axis=1)

    # PSD
    psd_features = band_features.mean(axis=1)

    # 差异熵:DE
    variance = band_features.var(axis=1)
    de_features = 0.5 * np.log(2 * np.pi * np.e * variance)

    # 香农熵:SE
    epsilon = 1e-10
    # 归一化为概率分布
    prob_dist = band_features / (band_features.sum(axis=1, keepdims=True) + epsilon)
    # 计算香农熵
    se_features = -np.sum(prob_dist * np.log(prob_dist + epsilon), axis=1)

    return psd_features, de_features, se_features

#############时域特征提取模块##################
class down_sample(nn.Module):
    def __init__(self, inc, kernel_size, stride, padding):
        super(down_sample, self).__init__()
        # 不同尺度的时间卷积
        self.conv = nn.Conv2d(in_channels = inc, out_channels = inc, kernel_size = (1, kernel_size), stride = (1, stride), padding = (0, padding), bias = False)
        # 批归一化
        self.bn = nn.BatchNorm2d(inc)
        self.elu = nn.ELU(inplace = False)
        # 自定义权重初始化规则
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 检查m是否为Conv2d实例
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.elu(self.bn(self.conv(x)))
        return output

class input_layer(nn.Module):
    def __init__(self, outc, groups):
        super(input_layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=outc, kernel_size = (1,3),
                                    stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn_input = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)
        self.initialize()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): # x: [batch_size, 1, eeg_channels, timepoints]
        output = self.elu(self.bn_input(self.conv_input(x)))
        return output # [batch_size, outc, eeg_channels, timepoints]

class Residual_Block(nn.Module):
    def __init__(self, inc, outc, groups = 1):
        super(Residual_Block, self).__init__()
        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1,
                                         stride=1, padding=0, groups=groups, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(1, 3),
                               stride=1, padding=(0, 1), groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=(1, 3),
                               stride=1, padding=(0, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): # x: [batch_size, inc, eeg_channels, timepoints]
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.bn2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output # [batch_size, outc, eeg_channels, timepoints]

def embedding_network(input_layer, Residual_Block, num_of_layer, outc, groups = 1):
    layers = []
    layers.append(input_layer(outc, groups=groups))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), outc = int(math.pow(2, i+1)*outc), groups = groups))
    return nn.Sequential(*layers)

class Multi_Scale_Temporal_Block(nn.Module):
    def __init__(self, outc, num_of_layer = 1):
        super().__init__()
        self.num_of_layers = num_of_layer
        self.embedding = embedding_network(input_layer, Residual_Block, num_of_layer=num_of_layer, outc=outc, groups=1)
        # 下面的downsample模块 输入:[batch_size, 5, eeg_channels, timepoints] 输出:[batch_size, 5, eeg_channels, timepoints_out]
        # timepoints_out = (timepoints + 2*padding - kernel_size) / stride + 1
        self.downsample1 = down_sample(outc*int(math.pow(2, num_of_layer))+1, kernel_size=4, stride=4, padding=0)
        self.downsample2 = down_sample(outc*int(math.pow(2, num_of_layer))+1, kernel_size=8, stride=8, padding=0)
        self.downsample3 = down_sample(outc*int(math.pow(2, num_of_layer))+1, kernel_size=16, stride=16, padding=0)
        self.downsample4 = down_sample(outc*int(math.pow(2, num_of_layer))+1, kernel_size=32, stride=32, padding=0)
        self.downsample5 = down_sample(outc*int(math.pow(2, num_of_layer))+1, kernel_size=32, stride=32, padding=0)

    def forward(self, x): # x: [batch_size, 1, eeg_channels, timepoints]
        embedding_x = self.embedding(x) # [batch_size, 4, eeg_channels, timepoints] outc = 2, num_of_layer = 1
        # 拼接原始的eeg和经过残差网络的eeg，在特征图通道层面拼接
        cat_x = torch.cat((embedding_x, x), 1) # [batch_size, 5, eeg_channels, timepoints]
        downsample1 = self.downsample1(cat_x) # [batch_size, 5, eeg_channels, timepoints_out1]
        downsample2 = self.downsample1(cat_x) # [batch_size, 5, eeg_channels, timepoints_out2]
        downsample3 = self.downsample1(cat_x) # [batch_size, 5, eeg_channels, timepoints_out3]
        downsample4 = self.downsample1(cat_x) # [batch_size, 5, eeg_channels, timepoints_out4]
        downsample5 = self.downsample1(cat_x) # [batch_size, 5, eeg_channels, timepoints_out5]
        temporal_features = torch.concat((downsample1, downsample2, downsample3, downsample4, downsample5), 3)
        # [batch_size, 5, eeg_channels, timepoints_out1+timepoints_out2+timepoints_out3+timepoints_out4+timepoints_out5]
        return temporal_features


#############图卷积模块##################
# 自适应邻接矩阵构建
class graph_constructor(nn.Module):
    def __init__(self, dim, k, device, alpha=3): # dim:embedding每个通道的特征数 k:稀疏化保留k个邻居节点
        super(graph_constructor, self).__init__()
        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)
        self.device = device
        self.dim = dim
        self.alpha = alpha
        self.k = k

    def forward(self, emb): # emb:图嵌入特征 [batch_size, eeg_channels, features]
        M1 = torch.tanh(self.alpha*self.lin1(emb))
        M2 = torch.tanh(self.alpha*self.lin2(emb))

        a = torch.bmm(M1, M2.transpose(2,1))-torch.bmm(M2, M1.transpose(2,1))
        adj = F.tarelu(torch.tanh(self.alpha*a)) # [batch_size, eeg_channels, eeg_channels]
        mask = torch.zeros_like(adj).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,2)
        mask.scatter_(2,t1,s1.fill_(1))
        adj = adj*mask
        return adj # [batch_size, eeg_channels, eeg_channels]

# 图卷积层
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)
class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,beta): # c_in: embedding输入前的每个通道特征数 # c_in: embeding输出后的每个通道特征数
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep # GNN深度
        self.beta = beta


    def forward(self,x,adj): # x:embedding特征[batch_size, eeg_channels, features] # adj:[batch_size, eeg_channels, eeg_channels]
        adj = adj + torch.eye(adj.size(1)).to(x.device) # 添加自环:A+I
        d = adj.sum(2) # 计算节点度，d为度向量(出度？) D_ii = 1+sum(j)A_ij  [batch_size, eeg_channels]
        h = x
        out = [h]
        a = adj / d.view(adj.size(0), adj.size(1), 1) # 邻接矩阵度归一化:A^hat = D^{-1}(A+I) [batch_size, eeg_channels, eeg_channels]
        for i in range(self.gdep):
            # h = self.beta*x + (1-self.beta)*self.nconv(h,a) # self.nconv(h,a):A^hat*H^{k-1}
            h = self.beta * x + (1 - self.beta) * torch.bmm(a, h)  # self.nconv(h,a):A^hat*H^{k-1} [batch_size, eeg_channels, eeg_channels]
            out.append(h)
        ho = torch.cat(out,dim=2) # [batch_size, eeg_channels, (gdep+1)*eeg_channels]
        ho = self.mlp(ho) # [batch_size, eeg_channels, eeg_channels]
        return ho

# 我实现的方法1
    # def forward(self,x,adj):
    #     adj = adj + torch.eye(adj.size(0)).to(x.device) # 添加自环:A+I
    #     d_out = torch.pow(adj.sum(1), 0.5)
    #     d_in = torch.pow(adj.sum(0), 0.5)
    #     h = x
    #     out = [h]
    #     a = (adj / d_out.view(-1, 1)) / d_in
    #     for i in range(self.gdep):
    #         h = self.alpha*x + (1-self.alpha)*self.nconv(h,a) # self.nconv(h,a):A^hat*H^{k-1}
    #         out.append(h)
    #     ho = torch.cat(out,dim=1)
    #     ho = self.mlp(ho)
    #     return ho
# 我实现的方法2
#     def forward(self, x, adj):
#         adj = adj + torch.eye(adj.size(0)).to(x.device) # 添加自环:A+I
#         d_out = torch.pow(adj.sum(1), -0.5)
#         d_in = torch.pow(adj.sum(0), -0.5)
#
#         # 处理无穷大值（度为零的情况）
#         d_out[torch.isinf(d_out)] = 0
#         d_in[torch.isinf(d_in)] = 0
#
#         # 创建对角矩阵
#         d_out = torch.diag(d_out)
#         d_in = torch.diag(d_in)
#
#         #　D_{out}^{ -\frac{1}{2}}AD_{in}^{ -\frac{1}{2}}
#         a = torch.mm(torch.mm(d_out, adj), d_in)
#
#         h = x
#         out = [h]
#         for i in range(self.gdep):
#             h = self.alpha * x + (1 - self.alpha) * self.nconv(h,a) # 图卷积层
#             out.append(h)
#         ho = torch.cat(out,dim=1)
#         ho = self.mlp(ho)
#         return ho


# 最终模型
# class Adopt_Gnn_PSD_DE_SE(nn.Module):
#     def __init__(self, num_classes, dim, k, device, alpha_PSD, alpha_DE, alpha_SE):
#         super().__init__()
#         # self.time_block = Multi_Scale_Temporal_Block(outc=2)
#         # self.femap_compress = nn.Conv2d(in_channels = 5, out_channels = 1, kernel_size = (1, 3),
#         #                             stride = 1, padding = (0, 1),  bias = False)
#         self.gc_PSD = graph_constructor(dim, k, device, alpha=alpha_PSD)
#         self.gc_DE = graph_constructor(dim, k, device, alpha=alpha_DE)
#         self.gc_SE = graph_constructor(dim, k, device, alpha=alpha_SE)
#
#     def forward(self,x): # x: [batch_size, 1, eeg_channels, timepoints]
#         temporal_fe = self.time_block(x) # [batch_size, 5, eeg_channels, timepoints_outs]
#         embedding = self.femap_compress(temporal_fe) # [batch_size, 1, eeg_channels, timepoints_outs]
class Residual_gconv(nn.Module): # 可以考虑设置droupout
    def __init__(self, dim, gcn_depth, beta):
        super(Residual_gconv,self).__init__()
        self.gconv1 = mixprop(dim, dim, gcn_depth, beta)
        self.gconv2 = mixprop(dim, dim, gcn_depth, beta)

    def forward(self,x, adj): # x [batch_size, eeg_channels, features] adj [batch_size, eeg_channels, eeg_channels]
        h = self.gconv1(x, adj) + self.gconv1(x, adj.transpose(1,2)) # [batch_size, eeg_channels, features]
        h_out = x + h
        return h_out


class Adopt_Gnn_PSD_DE_SE(nn.Module):
    def __init__(self, dim, eeg_channels, subgraph_size, device, alpha_PSD, alpha_DE, alpha_SE, beta_PSD, beta_DE, beta_SE, gcn_depth, layers, num_classes=2):
        super().__init__()
        self.gc_PSD = graph_constructor(dim, subgraph_size, device, alpha=alpha_PSD)
        self.gc_DE = graph_constructor(dim, subgraph_size, device, alpha=alpha_DE)
        self.gc_SE = graph_constructor(dim, subgraph_size, device, alpha=alpha_SE)
        self.resgconv_PSD = nn.ModuleList()
        self.resgconv_DE = nn.ModuleList()
        self.resgconv_SE = nn.ModuleList()
        for i in range(1, layers + 1):
            self.resgconv_PSD.append(Residual_gconv(dim, gcn_depth, beta_PSD))
            self.resgconv_DE.append(Residual_gconv(dim, gcn_depth, beta_DE))
            self.resgconv_SE.append(Residual_gconv(dim, gcn_depth, beta_SE))
        self.skip_PSD = nn.Linear((layers + 1) * dim, dim)
        self.skip_DE = nn.Linear((layers + 1) * dim, dim)
        self.skip_SE = nn.Linear((layers + 1) * dim, dim)
        self.output = nn.Linear(eeg_channels * 3 * dim, num_classes)
        self.layers = layers


#　外层的大图卷积层(layers=3)也可以参考残差设计
    def forward(self,x): # x: [batch_size, 3, eeg_channels, features]
        x_PSD = x[:, 0, :, :] # [batch_size, eeg_channels, features]
        x_DE = x[:, 1, :, :]
        x_SE = x[:, 2, :, :]
        adj_PSD = self.gc_PSD(x_PSD)
        adj_DE = self.gc_DE(x_DE)
        adj_SE = self.gc_SE(x_SE)
        h_PSD = [x_PSD]
        h_DE = [x_DE]
        h_SE = [x_SE]
        for i in range(self.layers):
            x_PSD = self.resgconv_PSD[i](x_PSD, adj_PSD) # [batch_size, eeg_channels, features]
            h_PSD.append(x_PSD)
            x_DE = self.resgconv_DE[i](x_DE, adj_DE)
            h_DE.append(x_DE)
            x_SE = self.resgconv_SE[i](x_SE, adj_SE)
            h_SE.append(x_SE)
        h_PSD = torch.cat(h_PSD, dim=2) # [batch_size, eeg_channels, (layer + 1) * features]
        h_DE = torch.cat(h_DE, dim=2)
        h_SE = torch.cat(h_SE, dim=2)

        h_PSD = self.skip_PSD(h_PSD) # [batch_size, eeg_channels, features]
        h_DE = self.skip_DE(h_DE)
        h_SE = self.skip_SE(h_SE)

        h_out = torch.cat((h_PSD, h_DE, h_SE), dim=2) # [batch_size, eeg_channels, 3 * features]
        # h_out = self.output(h_out)
        h_out = h_out.reshape(h_out.size(0), -1)  # [batch_size, (eeg_channels) * (3 * features)]
        h_out = self.output(h_out)
        return h_out


