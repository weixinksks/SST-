import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from copy import deepcopy


def runmean(data, n_run):
    ll = data.shape[0]
    data_run = np.zeros([ll])
    for i in range(ll):
        if i < (n_run - 1):
            data_run[i] = np.nanmean(data[0: i + 1])
        else:
            data_run[i] = np.nanmean(data[i - n_run + 1: i + 1])
    return data_run


def cal_ninoskill2(pre_nino_all, real_nino):
    """
    :param pre_nino_all: [n_yr,start_mon,lead_max]
    :param real_nino: [n_yr,12]
    :return: nino_skill: [12,lead_max]
    """
    lead_max = pre_nino_all.shape[2]
    nino_skill = np.zeros([12, lead_max])
    for ll in range(lead_max):
        lead = ll + 1
        dd = deepcopy(pre_nino_all[:, :, ll])
        for mm in range(12):
            bb = dd[:, mm]
            st_m = mm + 1
            terget = st_m + lead
            if 12 < terget <= 24:
                terget = terget - 12
            elif terget > 24:
                terget = terget - 24
            aa = deepcopy(real_nino[:, terget - 1])
            nino_skill[mm, ll] = np.corrcoef(aa, bb)[0, 1]
    return nino_skill


def compute_land_ratio(data):
    """
    需要计算每个patch中陆地区域的比例，从而调整权重
    Compute the ratio of land in a given flattened patch.
    Args:
    - flattened_patch (numpy array): 1D array representing a flattened patch.

    Returns:
    - ratio (float): ratio of land in the patch.
    """
    # 1.计算得到每个patch的l2范数，之后进行归一化从而得到每个patch的重要比例，进一步调整注意力机制的关注区域
    l2_norms = np.linalg.norm(data.cpu(), axis=-1)
    # 2. 计算所有patch的L2范数之和
    # total_norm = np.sum(l2_norms)
    # # 3. 计算每个patch的L2范数在所有patch的总和比例
    # land_ratio = l2_norms / total_norm
    # print(land_ratio)
    # print(land_ratio.shape)

    land_ratio = torch.from_numpy(l2_norms)
    # land_ratio = torch.from_numpy(land_ratio)
    return land_ratio


# class make_embedding(nn.Module):
#     def __init__(
#             self,
#             cube_dim,
#             d_size,
#             emb_spatial_size,
#             max_len,
#             device,
#     ):
#         """
#         :param cube_dim: The number of grids in one patch cube
#         :param d_size: the embedding length
#         :query, key, value: size (batch, S, T, d_size)
#         :param max_len: look back or prediction length, T
#         """
#         super().__init__()
#         # 1. temporal embedding
#         pe = torch.zeros(max_len, d_size)
#         temp_position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_size, 2) * -(np.log(10000.0) / d_size))
#         pe[:, 0::2] = torch.sin(temp_position * div_term)
#         pe[:, 1::2] = torch.cos(temp_position * div_term)
#         self.pe_time = pe[None, None].to(device)
#         # 2. spatial embedding
#         # 生成固定的位置标识
#         self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
#         # 定义一个可以学习的参数矩阵，当模型输入一个空间位置标识符时，nn.Embedding 会查找并返回对应的空间特征向量。
#         # 这个特征向量可以被视为该空间位置的表示，其中包含了与该位置相关的特征信息
#         self.emb_space = nn.Embedding(emb_spatial_size, d_size)
#
#         # 层级
#         self.linear = nn.Linear(cube_dim, d_size)
#         self.norm = nn.LayerNorm(d_size)
#
#     def forward(self, x):
#         # 在这里将整体区域位置编码实现
#
#
#
#
#
#
#         assert len(x.size()) == 4
#         embedded_space = self.emb_space(self.spatial_pos)
#
#         # print(self.linear(x).shape)
#         # print(embedded_space.shape)
#         # print(self.pe_time[:, :, : x.size(2)].shape)
#         x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
#
#         return self.norm(x)

# 修改后      # 在这里将整体区域位置编码实现
class make_embedding(nn.Module):
    def __init__(
            self,
            cube_dim,
            d_size,
            emb_spatial_size,
            max_len,
            device,
    ):
        """
        :param cube_dim: The number of grids in one patch cube
        :param d_size: the embedding length
        :query, key, value: size (batch, S, T, d_size)
        :param max_len: look back or prediction length, T
        """
        super().__init__()
        # 1. space embedding
        pe = torch

        pe = torch.zeros(max_len, d_size)
        temp_position = torch.arange(0, max_len).unsqueeze(1)
        # print(temp_position.shape)
        div_term = torch.exp(torch.arange(0, d_size, 2) * -(np.log(10000.0) / d_size))
        # print(div_term.shape)
        pe[:, 0::2] = torch.sin(temp_position * div_term)
        pe[:, 1::2] = torch.cos(temp_position * div_term)
        self.pe_time = pe[None, None].to(device)
        # 2. spatial embedding
        # 生成固定的位置标识
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
        # 定义一个可以学习的参数矩阵，当模型输入一个空间位置标识符时，nn.Embedding 会查找并返回对应的空间特征向量。
        # 这个特征向量可以被视为该空间位置的表示，其中包含了与该位置相关的特征信息
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)

        # 层级
        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size)

    def forward(self, x):
        # 在这里将整体区域位置编码实现

        # assert len(x.size()) == 4
        embedded_space = self.emb_space(self.spatial_pos)
        x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
        return self.norm(x)


def unfold_func(in_data, kernel_size, patch_steps):
    n_dim = len(in_data.size())
    assert n_dim == 4 or n_dim == 5
    # 这里直接切分为非重叠固定区域大小
    # 改进点:是否可以将非重叠区域切分->重叠区域切分,这里可以直接将步长设置为其他的大小
    data1 = in_data.unfold(-2, size=kernel_size[0], step=patch_steps)
    # print(data1.shape)
    # 经过unfold之后,data1的形状由[4,2,70,161,161]->[4,2,70,23,161,7]
    data1 = data1.unfold(-2, size=kernel_size[1], step=patch_steps).flatten(-2)
    if n_dim == 4:
        data1 = data1.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:
        data1 = data1.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert data1.size(-3) == in_data.size(-3) * kernel_size[0] * kernel_size[1]
    return data1


def fold_func(tensor, output_size, patch_size,stride):
    tensor = tensor.float()
    # print(tensor.shape)
    n_dim = len(tensor.size())
    # print(n_dim)
    assert n_dim == 5 or n_dim == 6
    # 需要判断数据维度大小,从而实现数据的重塑

    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    # print(f.shape)
    # f = f.flatten(0, 1) if n_dim == 5 else f
    # print(f.shape)
    folded = F.fold(
        f.flatten(-2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=stride,
    )
    # print(folded.shape)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), 1,*folded.size()[2:])
    return folded


def clone_layer(layer_in, N):
    return nn.ModuleList([copy.deepcopy(layer_in) for _ in range(N)])


# 进行层级连接的作用
class layerConnect(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# 时间注意力机制
def T_attention(query, key, value, dropout=None, mask_bias=None):
    d_k = query.size(-1)
    sc = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    # print(sc)
    # print(sc.shape)
    # if mask is not None:
    #     assert mask.dtype == torch.bool
    #     assert len(mask.size()) == 2
    #     sc = sc.masked_fill(mask[None, None, None], float("-inf"))
    p_sc = F.softmax(sc, dim=-1)

    # print(p_sc.shape)
    if mask_bias is not None:
        # print(mask_bias.shape)
        mask_bias = mask_bias.unsqueeze(1).unsqueeze(-1)
        # print(mask_bias.shape)
        p_sc = p_sc * mask_bias
    # print(p_sc)
    # print(p_sc.shape)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value)


# 空间注意力机制
# 需要进行转置
def S_attention(query, key, value, dropout=None, mask_bias=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query.transpose(2, 3), key.transpose(2, 3).transpose(-2, -1)
    ) / np.sqrt(d_k)
    # print(scores)
    # print(scores.shape)
    p_sc = F.softmax(scores , dim=-1)
    # print(p_sc.shape)
    # 进行元素相乘，调整patch的权重分布
    if mask_bias is not None:
        # 将掩码偏置进行转置
        mask_bias = mask_bias.transpose(-2, -1)
        # print(mask_bias)
        # print(mask_bias.shape)
        mask_bias = mask_bias.unsqueeze(1).unsqueeze(-1)
        p_sc = p_sc * + mask_bias
    # print(p_sc)
    # print(p_sc.shape)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value.transpose(2, 3)).transpose(2, 3)


class make_attention(nn.Module):
    def __init__(self, d_size, nheads, attention_module, dropout):
        super().__init__()
        assert d_size % nheads == 0
        self.d_k = d_size // nheads
        self.nheads = nheads
        # 设置多少层的模型
        self.linears = nn.ModuleList([nn.Linear(d_size, d_size) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attention_module = attention_module

    def forward(self, query, key, value, mask_bias):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (batch, S, T, d_size)
        Returns:
            (batch, S, T, d_size)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        # 通过这个方法来获得得到相关的Q，K，V
        query, key, value = [
            l(x)
            .view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 设置相关的注意力
        # print(mask_bias)
        # print(mask_bias.shape)

        x = self.attention_module(query, key, value, mask_bias=mask_bias, dropout=self.dropout)

        x = (
            x.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        )
        return self.linears[-1](x)


class miniEncoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clone_layer(layerConnect(size=d_size, dropout=dropout), 2)

        # 制作时间注意力机制
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        # 制作空间注意力机制
        self.space_attn = make_attention(d_size, nheads, S_attention, dropout)

        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.LeakyReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    # 进行时空注意力机制融合
    def TS_attn(self, query, key, value, mask_bias):
        """
        Returns:
            (batch, S, T, d_size)
        """
        tt = self.time_attn(query, key, value, mask_bias)
        return self.space_attn(tt, tt, tt, mask_bias)

    def forward(self, x, mask_bias):
        # print(mask_bias)
        # print(mask_bias.shape)
        x = self.sublayer[0](x, lambda x: self.TS_attn(x, x, x, mask_bias))
        return self.sublayer[1](x, self.FC)


class miniDecoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clone_layer(layerConnect(d_size, dropout), 2)
        self.encoder_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.space_attn = make_attention(d_size, nheads, S_attention, dropout)
        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.LeakyReLU(),
            nn.Linear(dim_feedforward, d_size)
        )

    def divided_TS_attn(self, query, key, value, mask_bias=None):
        m = self.time_attn(query, key, value, mask_bias)
        return self.space_attn(m, m, m,mask_bias)

    def forward(self,x, mask_bias):

        # print(x.shape)
        # print(mask_bias.shape)
        x = self.sublayer[0](x, lambda x: self.divided_TS_attn(x, x, x, mask_bias))
        
        # 这里需要去去除计算交叉注意力，由于模型的输入改变为单输入，因此无法实现交叉注意力的计算，故此需要去掉
        # x = self.sublayer[1](
        #     x, lambda x: self.encoder_attn(x, en_out, en_out, memory_mask)
        # )


        return self.sublayer[1](x, self.FC)
