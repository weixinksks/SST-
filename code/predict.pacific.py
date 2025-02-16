from Geoformer import Geoformer
import numpy as np
import xarray as xr
from myconfig import mypara
import torch
import os
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def norm_data(norm_func, data):
    ngroup, time, channel, lat, lon = data.shape
    # print(ngroup, time, lat, lon)
    # # 将数据转换为适合归一化的形状,并进行归一化，之后进行重塑
    data = data.reshape(ngroup * time * channel, lat, lon)
    for i in range(data.shape[0]):
        print(i)
        # 首先计算二维矩阵非nan值的均值
        # mean_data = np.nanmean(data[i])
        # 之后进行填充nan
        # data[i] = np.nan_to_num(data[i],nan=mean_data)
        # imputer = SimpleImputer(strategy='constant', fill_value=0)
        # filled_data = imputer.fit_transform(data[i])
        # print("填充之后的数据:",filled_data)
        # 将填充之后的数据进行归一化
        data[i] = torch.from_numpy(norm_func.fit_transform(data[i]))
    data = data.reshape(ngroup, time, channel, lat, lon)
    return data


def inversenorm_data(norm_func, data):
    ngroup, time, channel, lat, lon = data.shape
    # # 将数据转换为适合归一化的形状,并进行归一化，之后进行重塑
    data = data.reshape(ngroup * time * channel, lat, lon)
    for i in range(data.shape[0]):
        # print("现在正在将数据反归一化：",i)
        # print("归一化之后的数据：",data[i])
        data[i] = torch.from_numpy(norm_func.inverse_transform(data[i]))
        # print("反归一化之后的数据：",data[i])
    data = data.reshape(ngroup, time, channel, lat, lon)
    return data


start_index = 0
end_index = 4

# 对于数据进行归一化,需要保证跟训练集的归一化方法一致。一致采用StandardScaler方法
model_norm_func = StandardScaler()
error_norm_func = StandardScaler()

# # 设置经度和纬度的间隔
lon_interval = 2
lat_interval = 2

# 预测数据地址  在这里读入预测数据
# data_in = xr.open_dataset(mypara.predict_model)

data_in = xr.open_dataset(mypara.predict_error)

lat = data_in["lat"].values
lon = data_in["lon"].values
lon = lon[::lon_interval]
lat = lat[::lat_interval]

lon_range = mypara.lon_range
lat_range = mypara.lat_range
input_length = mypara.input_length
output_length = mypara.output_length
train_length = mypara.train_length
eval_length = mypara.eval_length
all_group = mypara.all_group

field_data_in = data_in['err'].values  # 跟load data的变量更改为一致

# 使用 NumPy 中的函数实现降低分辨率
field_data_in = field_data_in[:, start_index, ::lon_interval, ::lat_interval]
field_data_in = np.where(np.isnan(field_data_in), 0, field_data_in)

field_data_in = np.expand_dims(field_data_in, axis=1)
field_data_in = np.expand_dims(field_data_in, axis=2)

# field_data_in = np.transpose(field_data_in, [1, 0, 2, 3])
# field_data_in = field_data_in[:, :1, :, :]

data_out = xr.open_dataset(mypara.predict_error)
print(data_out)
outtime = data_out['time']
print(outtime)
field_data_out = data_out['err'].values
field_data_out = field_data_out[:, end_index, ::lon_interval, ::lat_interval]

nsample = field_data_out.shape[0]
print('23年的所有样本', nsample)
field_data_out = np.where(np.isnan(field_data_out), 0, field_data_out)
field_data_out = np.expand_dims(field_data_out, axis=1)
field_data_out = np.expand_dims(field_data_out, axis=2)
# field_data_out = np.transpose(field_data_out, [0, 2, 1, 3, 4])
# field_data_out = field_data_out[:, :, :1, : ,:]
print(field_data_in.shape, field_data_out.shape)

modellist = sorted(glob.glob(
    f'/home/ysp/zgq/transformer/SST_zhangv2/model/Geoformer.pacific.adam.Huber.1.9_9.4_4_4_end_index4_huber1.pkl'))
for model_path in modellist:
    print(model_path)
    model_name = model_path.split('/')[-1]
    region = model_name.split('.')[1]
    active = model_name.split('.')[2]
    loss = model_name.split('.')[3]
    input_length = int(model_name.split('.')[4])
    hw = model_name.split('.')[5]
    layer = model_name.split('.')[6]

    os.system(
        f'mkdir -p /home/ysp/zgq/transformer/SST_zhangv2/predict_output/{active}.{loss}.{input_length}.{hw}.{layer}.{end_index}')

    mymodel = Geoformer(mypara).to(mypara.device)

    print("模型的地址", model_path)

    loaded_model = torch.load(model_path, map_location=mypara.device)
    # # print("模型字典",state_dict)

    # 提取模型参数
    state_dict = loaded_model.state_dict()
    #
    mymodel.load_state_dict(state_dict)
    # mymodel = torch.load(model_path)
    # mymodel = torch.load(model_path, map_location=mypara.device)
    # mymodel = mymodel.to('cpu')
    # mydevice = mymodel.device
    # print(mydevice)
    # mydevice = 'cuda:3'
    mymodel = mymodel.to(mypara.device)
    mydevice = mymodel.device
    print(mydevice)

    # mydevice = 'cpu'
    mymodel.eval()

    with torch.no_grad():
        for i in range(nsample):  # 遍历输出数据集中的每个时间步

            # _time = outtime[i].values
            # print(_time)
            # _time = datetime.strftime(pd.to_datetime(_time), '%Y-%m-%d %H:%M:%S')
            # print('时间分割之后', _time)
            # yr = _time.split(' ')[0].split('-')[0]
            # mn = _time.split(' ')[0].split('-')[1]
            # dy = _time.split(' ')[0].split('-')[2]
            # hr = _time.split(' ')[1].split(':')[0]
            # print(i)
            # print("field_data_in of shape : ", field_data_in.shape)
            # print("field_data_out of shape : ", field_data_out.shape)

            input_var = field_data_in[i]
            input_var = np.expand_dims(input_var, axis=0)
            true_var = field_data_out[i]
            true_var = np.expand_dims(true_var, axis=0)
            input_var = torch.tensor(input_var, dtype=torch.float32)
            true_var = torch.tensor(true_var, dtype=torch.float32)

            # print("input_var of shape ", input_var.shape)

            # 需要在这里将模式数据进行归一化
            # input_var = norm_data(model_norm_func, input_var.cpu())
            # true_var = norm_data(error_norm_func, true_var.cpu())

            out_var = mymodel(
                input_var.to(mydevice),
                train=False,
            )
            # print("打印出预测的值", out_var)
            # print("打印出预测的形状", out_var.shape)

            # # 预测结果出来直接进行反归一化
            # out_var = inversenorm_data(error_norm_func, out_var.cpu())
            # true_var = inversenorm_data(error_norm_func, true_var.cpu())

            out_var = out_var.squeeze()
            true_var = true_var.squeeze()
            #
            # print("out_var shape ", out_var.shape)
            # print("true_var shape ", true_var.shape)

            out_var = xr.DataArray(out_var.cpu().detach().numpy(), dims=['lat', 'lon'],
                                   coords=[lat, lon])
            out_var.name = 'value'
            out_var.to_netcdf(
                f'/home/ysp/zgq/transformer/SST_zhangv2/predict_output/{active}.{loss}.{input_length}.{hw}.{layer}.{end_index}/predict_{i}.nc')

            true_var = xr.DataArray(true_var, dims=['lat', 'lon'], coords=[lat, lon])
            true_var.name = 'value'
            true_var.to_netcdf(
                f'/home/ysp/zgq/transformer/SST_zhangv2/predict_output/{active}.{loss}.{input_length}.{hw}.{layer}.{end_index}/error_{i}.nc')
