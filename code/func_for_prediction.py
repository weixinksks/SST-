from Geoformer import Geoformer
# from SST import make_dataset
import torch
from torch.utils.data import DataLoader


def inversenorm_data(norm_func,data):
    ngroup, time,channel ,lat, lon = data.shape
    # # 将数据转换为适合归一化的形状,并进行归一化，之后进行重塑
    data = data.reshape(ngroup * time*channel, lat, lon)
    for i in range(data.shape[0]):
        # print("现在正在将数据反归一化：",i)
        # print("归一化之后的数据：",data[i])
        data[i] =torch.from_numpy(norm_func.inverse_transform(data[i]))
        # print("反归一化之后的数据：",data[i])
    data = data.reshape(ngroup, time,channel, lat, lon)
    return data


# 定义一个预测函数，作用是对于测试数据进行预测
def func_pre(mypara, adr_model,eval_dataset,norm_func):
    # -------------
    # # 得到测试集的长度（记录测试集中输入模型的具体），同时打印测试集数据的形状
    # 制作输入样本（加上了batch_size信息）
    dataloader_test = DataLoader(
        eval_dataset, batch_size=mypara.batch_size_test, shuffle=True
    )
    # 加载一个构建的模型
    mymodel = Geoformer(mypara).to(mypara.device)
    # 将保存好的损失最小的模型参数加载到设定的mymodel中，这个过程会使得mymodel对象的参数与保存的模型参数完全一致，可以直接用于推断或微调等任务
    mymodel.load_state_dict(torch.load(adr_model))
    # 将模型切换到评估模式下，关闭dropout等操作
    mymodel.eval()
    # 使用训练好的模型进行推断，并将推断结果存储到var_pred中，表示该代码在不进行梯度计算的情况下执行
    with torch.no_grad():
        # 遍历dataloader_test测试中所有的输入数据
        for j, (input_var, var_true1) in enumerate(dataloader_test):
            # print("整个的归一化模式数据:",input_var)
            # print("整个的归一化误差数据:",var_true1)
            # output_mask = (var_true1 == 0)
            out_var = mymodel(
                input_var.float().to(mypara.device),
                train=False,
            )
            # 将预测数据和真实数据进行反归一化，恢复到原始数据的范围
            if norm_func is not None:
                out_var = inversenorm_data(norm_func,out_var.cpu())
                var_true1 = inversenorm_data(norm_func,var_true1.cpu())
            # out_var[output_mask] = 0
            yield (input_var,out_var, var_true1)
