import torch


class Mypara:
    def __init__(self):
        pass


mypara = Mypara()
mypara.device = torch.device("cuda:0")
mypara.batch_size_train = 1
mypara.batch_size_eval = 1
mypara.batch_size_test = 1
mypara.num_epochs = 300

mypara.TFnum_epochs = 20
mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.patience = 4
mypara.warmup = 300

mypara.interval = 4
mypara.TraindataProportion = 0.8
mypara.all_group = 13000
mypara.train_length = 0.8
mypara.eval_length = 0.9

mypara.needtauxy = False
# #
# mypara.adr_input = "../data/indian/sst.model.nc"
#
# mypara.adr_output = "../data/indian/sst.error.nc"



mypara.model = ("/home/ysp/zgq/transformer/SST_daily/sst.2018_2022.nc")

mypara.error = (
    "/home/ysp/zgq/transformer/SST_daily/err.2018_2022.nc"
)


mypara.predict_model = ("/home/ysp/zgq/transformer/SST_daily/sst.2023.nc")

mypara.predict_error = ("/home/ysp/zgq/transformer/SST_daily/err.2023.nc")



mypara.input_channal = 1  # n_lev of 3D temperature
# 输出的通道数
mypara.output_channal = 1
# 输入数据的历史时间长度,注意训练的长度是3，预测是输入长度是12
mypara.input_length = 1
# 输出的历史时间长度，注意训练的长度是3，预测是输入长度是12
mypara.output_length = 1

mypara.lon_range = (100, 145)
mypara.lat_range = (0, 45)



# # patch size，定义patch_size 的大小，太平洋
# mypara.patch_size = (8, 7)
# # 为了实现有重叠的切分
# mypara.patch_steps = 7


mypara.patch_size = (9, 9)

mypara.patch_steps = 6


# # 处理之后的数据不同的分辨率
mypara.lon_gridsize = 0.2
mypara.lat_gridsize = 0.2


# 在这里实际上需要去处理得到emb_spatial_size,可以使用网格数来进行修改定义，设置经度上的网格数，纬度上的网格数，从而来确定emb_spatial_szie
# 之后经纬度范围唯一的作用就是显示出真实经纬度区域
# 纬度，# 经度  太平洋
# mypara.lat_grids_range = (0, 136)
# mypara.lon_grids_range = (0, 154)
# # # # 印度洋
# mypara.lat_grids_range = (0, 90)
# mypara.lon_grids_range = (0, 140)

mypara.H0 = int(((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.lat_gridsize - mypara.patch_size[0])/ mypara.patch_steps +1)
mypara.W0 = int(((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.lon_gridsize - mypara.patch_size[1])/ mypara.patch_steps +1)
# 计算得到输入数据经过patchEmbedding得到的空间嵌入向量长度
mypara.emb_spatial_size = mypara.H0 * mypara.W0

# # 需要在这里进行处理分辨率的问题，这个在维度上进行三次切分，经度上进in行四次切分，如果采用[625,721]的话，
# # 计算输入数据在经度方向上的 patch 数量
# # 采用滑动方法来计算patch数量,计算方法如下: 这里可以设置相关的patch大小和步长
# mypara.H0 = int((mypara.lat_grids_range[1] - mypara.lat_grids_range[0] - mypara.patch_size[0]) / mypara.patch_steps + 1)
# # 同样是计算输入数在维度方向上的patch数量
# # mypara.W0 = int((mypara.lon_grids_range[1] - mypara.lon_grids_range[0]) / mypara.patch_size[1])
# mypara.W0 = int((mypara.lon_grids_range[1] - mypara.lon_grids_range[0] - mypara.patch_size[1]) / mypara.patch_steps + 1)
# # 计算得到输入数据经过patchEmbedding得到的空间嵌入向量长度
# mypara.emb_spatial_size = mypara.H0 * mypara.W0
# model
mypara.model_savepath = "./model/"
mypara.seeds = 1
mypara.d_size = 256
mypara.nheads = 4

mypara.dim_feedforward = 512
mypara.dropout = 0.2
mypara.num_encoder_layers = 4
mypara.num_decoder_layers = 4
