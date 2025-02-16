import os
import numpy as np
from torch.utils.data import Dataset
import xarray as xr
import random
import torch
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer


# from scipy.interpolate import interp1d
# from scipy.interpolate import griddata
# from impyute.imputation.cs import fast_knn
# from torch.nn.functional import interpolate
# from sklearn.impute import KNNImputer


class make_dataset(Dataset):
    def __init__(
            self,
            mypara,
            start_index,
            end_index,
            model_norm_func,
            error_norm_func,
            group

    ):
        self.mypara = mypara
        
        # 归一化方法-好像可以不使用
        self.model_norm_func = model_norm_func
        self.error_norm_func = error_norm_func
        
        # 采用0.5度的分辨率网格数据-模型输入err数据（0时刻），作为解码器输入，将后续的数据作为目标输出数据--先暂定不使用encoder
        
        # 这里默认数据是0.1分辨率数据，需要进行间隔采样
        # 设置经度和纬度的间隔
        lon_interval = 2
        lat_interval = 2

        
        # // 读取数据
        sst_data = xr.open_dataset(self.mypara.error)
        all_data = sst_data["err"].values
        
        # 获取0时刻的误差数据作为解码器的输入-需要增加一个维度符合模型输入要求
        decoder_in = all_data[:, start_index, ::lon_interval, ::lat_interval]
        decoder_in = np.expand_dims(decoder_in, axis=1)
        decoder_in = np.expand_dims(decoder_in, axis=2)
        decoder_in[np.isnan(decoder_in)] = 0
        print(np.isnan(decoder_in).sum())
        assert mypara.input_length == decoder_in.shape[1]
        
        
        # 获取后续的误差数据作为目标输出数据。
        # 这里的想法是每次训练一个模型，目标输出数据分别是1时刻，2时刻，3时刻，4时刻等的误差数据
        target = all_data[:, end_index, ::lon_interval, ::lat_interval]
        target = np.expand_dims(target, axis=1)
        target = np.expand_dims(target, axis=2)
        target[np.isnan(target)] = 0
        print(np.isnan(target).sum())
        assert mypara.output_length == target.shape[1]
        
        # 处理数据-使输入数据和输出数据一一对应
               # -----------------------------需要额外去处理这个，由于输入的数据包含了编码器输入，解码器输入，以及对应的目标输出
        self.decoder_data, self.target_data = self.deal_testdata(
             decoder_in=decoder_in, target=target, ngroup=group
        )
        
        
    def norm_data(self, norm_func, data):
        ngroup, time, channel, lat, lon = data.shape
        # print(ngroup, time, lat, lon)
        # # 将数据转换为适合归一化的形状,并进行归一化，之后进行重塑
        data = data.reshape(ngroup * time * channel, lat, lon)
        for i in range(data.shape[0]):
            # print(i)
            # 首先计算二维矩阵非nan值的均值
            mean_data = np.nanmean(data[i])
            # 之后进行填充nan
            # data[i] = np.nan_to_num(data[i],nan=mean_data)
            imputer = SimpleImputer(strategy='constant', fill_value=mean_data)
            filled_data = imputer.fit_transform(data[i])
            # print("填充之后的数据:",filled_data)
            # 将填充之后的数据进行归一化
            data[i] = norm_func.fit_transform(filled_data)
        data = data.reshape(ngroup, time, channel, lat, lon)
        return data

    def deal_testdata(self,  decoder_in, target, ngroup):
        # 这里需要定义关于目标输出的结果
        lb = decoder_in.shape[1]
        output_length = target.shape[1]
        # assert  lb == output_length
        if ngroup is None:
            ngroup = decoder_in.shape[0]

        out_field_decoder_in = np.zeros(
            [
                ngroup,
                lb,
                decoder_in.shape[2],
                decoder_in.shape[3],
                decoder_in.shape[4],
            ]
        )
        out_field_target = np.zeros(
            [
                ngroup,
                output_length,
                target.shape[2],
                target.shape[3],
                target.shape[4],
            ]
        )

        iii = 0
        for j in range(ngroup):
            rd = random.randint(0, decoder_in.shape[0] - 2)
            out_field_decoder_in[iii] = decoder_in[rd]
            out_field_target[iii] = target[rd]

            iii += 1

        return  out_field_decoder_in, out_field_target

    def getdatashape(self):
        return {
            "encoder_data": self.encoder_data.shape,
            "decoder_data": self.decoder_data.shape,
            "target_data":  self.target_data.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.mypara.lon_range[0],
                self.mypara.lon_range[1],
            ),
            "lat: {}S to {}N".format(
                -self.mypara.lat_range[0],
                self.mypara.lat_range[1],
            )
        }

    def __len__(self):
        return self.decoder_data.shape[0]

    def __getitem__(self, idx):
        return self.decoder_data[idx], self.target_data[idx]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        # # 设置经度和纬度的间隔
        # lon_interval = 2
        # lat_interval = 2

        # # 需要在这里添加uv场的数据，作为模型的编码器输入
        # u_data_in = xr.open_dataset(self.mypara.u)
        # encoder_in_u = u_data_in["u"].values
        # # u数据进行处理，填充，归一化等
        # encoder_in_u = encoder_in_u[:, :, ::lon_interval, ::lat_interval]
        # v_data_in = xr.open_dataset(self.mypara.v)
        # encoder_in_v = v_data_in["v"].values
        # # v数据进行处理，填充，归一化等
        # encoder_in_v = encoder_in_v[:, :, ::lon_interval, ::lat_interval]
        # # 将uv数据进行堆叠
        # encoder_in = np.stack((encoder_in_u, encoder_in_v), axis=2)
        # encoder_in[np.isnan(encoder_in)] = 0
        # print("encoder_in.shape",encoder_in.shape)
        # # 将数据进行堆叠之后,进行填充等操作
        # #encoder_in = self.norm_data(self.model_norm_func, encoder_in)
        # print("the shape of encoder_in after norm_func :",encoder_in.shape)

        # # 制作相关的模型输入数据，使得数据符合模型输入要求
        # data_in = xr.open_dataset(self.mypara.model)
        # # 直接获取数据中全部内容
        # decoder_in = data_in["sst"].values
        # mean = np.load('./lat.mean.npy')
        # decoder_in = decoder_in / mean * 10
        # # 使用 NumPy 中的函数实现降低分辨率
        # decoder_in = decoder_in[:, :, ::lon_interval, ::lat_interval]

        # # 需要对数据进行归一化，将模式数据的范围进行缩小，并且将误差数据也保持在相同的范围内，使得模型更好的捕捉对应的关系
        # # 在归一化之前使用均值填充nan值
        # # decoder_in = np.nan_to_num(decoder_in)
        # # decoder_in[abs(decoder_in) > 999] = 0
        # decoder_in = np.expand_dims(decoder_in, axis=2)
        # decoder_in[np.isnan(decoder_in)] = 0
        # print(np.isnan(decoder_in).sum())
        # #decoder_in = self.norm_data(self.model_norm_func, decoder_in)
        # assert mypara.input_length == decoder_in.shape[1]

        # # ================
        # data_out = xr.open_dataset(self.mypara.error)
        # target = data_out["err"].values
        # target = target[:, :, ::lon_interval, ::lat_interval]
        # target = np.expand_dims(target, axis=2)
        # target[np.isnan(target)] = 0
        # print(np.isnan(target).sum())
        # #target = self.norm_data(self.error_norm_func, target)
        # assert mypara.output_length == target.shape[1]

        # # -----------------------------需要额外去处理这个，由于输入的数据包含了编码器输入，解码器输入，以及对应的目标输出
        # self.encoder_data, self.decoder_data, self.target_data = self.deal_testdata(
        #     encoder_in=encoder_in, decoder_in=decoder_in, target=target, ngroup=None
        # )
        # print("self.encoder_data.shape:",self.encoder_data.shape)
        # print("self.decoder_data.shape:",self.decoder_data.shape)
        # print("self.target_data.shape:",self.target_data.shape)
        # # 由于数据定义之后,不想再被利用,可以删除
        # del encoder_in,decoder_in,target

    # def norm_data(self, norm_func, data):
    #     ngroup, time, channel, lat, lon = data.shape
    #     # print(ngroup, time, lat, lon)
    #     # # 将数据转换为适合归一化的形状,并进行归一化，之后进行重塑
    #     data = data.reshape(ngroup * time * channel, lat, lon)
    #     for i in range(data.shape[0]):
    #         # print(i)
    #         # 首先计算二维矩阵非nan值的均值
    #         mean_data = np.nanmean(data[i])
    #         # 之后进行填充nan
    #         # data[i] = np.nan_to_num(data[i],nan=mean_data)
    #         imputer = SimpleImputer(strategy='constant', fill_value=mean_data)
    #         filled_data = imputer.fit_transform(data[i])
    #         # print("填充之后的数据:",filled_data)
    #         # 将填充之后的数据进行归一化
    #         data[i] = norm_func.fit_transform(filled_data)
    #     data = data.reshape(ngroup, time, channel, lat, lon)
    #     return data

    # def deal_testdata(self, encoder_in, decoder_in, target, ngroup):
    #     # 这里需要定义关于目标输出的结果

    #     lb = decoder_in.shape[1]
    #     output_length = target.shape[1]
    #     # assert  lb == output_length
    #     if ngroup is None:
    #         ngroup = decoder_in.shape[0]

    #     out_field_encoder_in = np.zeros([
    #         ngroup,
    #         encoder_in.shape[1],
    #         encoder_in.shape[2],
    #         encoder_in.shape[3],
    #         encoder_in.shape[4]
    #     ])

    #     out_field_decoder_in = np.zeros(
    #         [
    #             ngroup,
    #             lb,
    #             decoder_in.shape[2],
    #             decoder_in.shape[3],
    #             decoder_in.shape[4],
    #         ]
    #     )
    #     out_field_target = np.zeros(
    #         [
    #             ngroup,
    #             output_length,
    #             target.shape[2],
    #             target.shape[3],
    #             target.shape[4],
    #         ]
    #     )
    #     iii = 0
    #     for j in range(ngroup):
    #         rd = random.randint(0, decoder_in.shape[0] - 2)
    #         # out_field_decoder_in[iii] = target[rd]
    #         # out_field_target[iii] = target[rd+1]
    #         out_field_encoder_in[iii] = encoder_in[rd]
    #         out_field_decoder_in[iii] = decoder_in[rd]
    #         out_field_target[iii] = target[rd]

    #         iii += 1
    #     # # print(out_field_decoder_in)
    #     # print(out_field_decoder_in.shape)
    #     # # print(out_field_target)
    #     # print(out_field_target.shape)

    #     return out_field_encoder_in, out_field_decoder_in, out_field_target

    # def getdatashape(self):
    #     return {
    #         "encoder_data": self.encoder_data.shape,
    #         "decoder_data": self.decoder_data.shape,
    #         "target_data":  self.target_data.shape,
    #     }

    # def selectregion(self):
    #     return {
    #         "lon: {}E to {}E".format(
    #             self.mypara.lon_range[0],
    #             self.mypara.lon_range[1],
    #         ),
    #         "lat: {}S to {}N".format(
    #             -self.mypara.lat_range[0],
    #             self.mypara.lat_range[1],
    #         )
    #     }

    # def __len__(self):
    #     return self.decoder_data.shape[0]

    # def __getitem__(self, idx):
    #     return self.encoder_data[idx], self.decoder_data[idx], self.target_data[idx]
