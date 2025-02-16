from Geoformer import Geoformer
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from SST import make_dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 定义学习率调整器，用于调整优化器中的学习率。
class lrwarm:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class modelTrainer:
    def __init__(self, mypara):
        # 判断输入和输出的数据通道是否相同
        # assert mypara.input_channal == mypara.output_channal
        # 获取相关的配置参数
        self.mypara = mypara
        # 获取配置的GPU地址
        self.device = mypara.device
        # 获取设置的transformer模型,
        self.mymodel = Geoformer(mypara).to(mypara.device)
        # 设置优化器，采用adam 的优化器
        # self.adam = torch.optim.Adam(self.mymodel.parameters(), lr=5e-5)
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        # 这行代码的作用是计算学习率的初始值，它通常用于优化算法中的梯度下降过程。这个学习率的初始值是基于一些参数的计算结果得出的。
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0001
        # 初始化一个优化器对象-有想法是利用Gan来进行生成
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)

        self.Huber_loss = torch.nn.SmoothL1Loss()

        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2

        # 创建一个张量，并将其发送到特定的设备中，设置一个nino的权重矩阵
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 8 + [4] * 6)
            * np.log(np.arange(25) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]

    def loss_var(self, y_pred, y_true, method):
        if method == 'rmse':
            rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
            rmse = rmse.sqrt().mean(dim=0)
            rmse = torch.sum(rmse, dim=[0, 1])
        if method == 'mae':
            rmse = torch.mean(torch.abs(y_pred - y_true), dim=[3, 4])
            rmse = rmse.sqrt().mean(dim=0)
            rmse = torch.sum(rmse, dim=[0, 1])
        if method == 'allrmse':
            rmse = torch.sum((y_pred - y_true) ** 2, dim=[0, 1, 2, 3, 4])
        if method == 'allmae':
            rmse = torch.sum(torch.abs(y_pred - y_true), dim=[0, 1, 2, 3, 4])
        if method == 'KL':
            p = torch.clamp(y_pred, min=1e-10)
            q = torch.clamp(y_true, min=1e-10)
            rmse = torch.sum(p * torch.log(p / q))
        if method == 'Huber':
            errors = y_pred - y_true
            abs_errors = torch.abs(errors)
            quadratic = torch.clamp(abs_errors, max=0.5)  # 0.5,1,2  更换huber 1 只是将这个max进行更换就可以了
            linear = (abs_errors - quadratic)
            rmse = 0.5 * quadratic ** 2 + 0.5 * linear
            rmse = rmse.mean()
        return rmse

    def L_2_loss(self, y_pred, y_true):
        l2_difference = torch.norm(y_pred - y_true, p=2)
        return l2_difference

        # 模型的预测部分

    # def model_pred(self, dataloader,method):
    #
    #     self.mymodel.eval()
    #     LOSS_VAR = []
    #     with torch.no_grad():
    #         for j, (input_var, var_true1) in enumerate(dataloader):
    #             output_mask = (var_true1 == 0)
    #             out_var = self.mymodel(
    #                 input_var.float().to(self.device),
    #                 train=False
    #             )
    #             out_var[output_mask] = 0
    #             loss_var = self.loss_var(out_var, var_true1.float().to(self.device))
    #             # loss_var = self.Huber_loss(out_var, var_true1.float().to(self.device))
    #             LOSS_VAR.append(loss_var)
    #             print("-->Evaluation... loss_var:{:.3f} ".format(
    #                 loss_var
    #             ))
    #         # 直接计算损失均值
    #         print(torch.mean(torch.tensor(LOSS_VAR)))
    #         LOSS_VAR = sorted(LOSS_VAR, reverse=True)
    #         # 获得评估数据集中的每个样本的评估损失之后，需要计算所有评估损失的平均值，从而来判断是否需要记录模型
    #         average_loss = sum(LOSS_VAR) / len(LOSS_VAR)
    #
    #     return (
    #         average_loss,
    #     )

    def model_pred(self, dataloader, method):
        self.mymodel.eval()
        var_pred = []
        var_true = []
        with torch.no_grad():
            for  decoder_input_var, var_true1 in dataloader:
                output_mask = (var_true1 == 0)
                out_var = self.mymodel(
                    decoder_input_var.float().to(self.device),
                    train=False,
                )
                out_var[output_mask] = 0
                var_true.append(var_true1)
                var_pred.append(out_var)
            var_pred = torch.cat(var_pred, dim=0)
            var_true = torch.cat(var_true, dim=0)
            # --------------------
            loss_var = self.loss_var(var_pred, var_true.float().to(self.device), method).item()
        return (
            var_pred,
            loss_var,
        )

    # 模型的训练部分
    def train_model(self, dataset_train, dataset_eval, method):
        chk_path = self.mypara.model_savepath + f"Geoformer.pacific.adam.{method}.{self.mypara.input_length}.{self.mypara.patch_size[0]}_{self.mypara.patch_size[1]}.{self.mypara.nheads}_{self.mypara.num_encoder_layers}_{self.mypara.num_decoder_layers}_end_index4_huber0.5.pkl"
        torch.manual_seed(self.mypara.seeds)
        # 加载训练数据
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=True
        )
        # 加载评估数据
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=True
        )
        count = 0
        # 记录最小值，这里表示为最小值，负无穷，需要更改为最大值，即正无穷
        best = math.inf
        print("the firstly best value of loss is {}".format(best))
        # 进行半监督学习的部分，首先将预测内容一部分进行替换为观测数据，0为无监督，1为全监督
        sv_ratio = 1
        for i_epoch in range(self.mypara.num_epochs):
            print("==========" * 8)
            print("\n-->epoch: {0}".format(i_epoch))
            # -------# -train # 开始训练
            self.mymodel.train()
            # 开始遍历数据
            for j, (decoder_input_var, var_true) in enumerate(dataloader_train):
                # print("误差数据的形状是{}".format(var_true.shape))
                # 清除梯度
                self.opt.optimizer.zero_grad()
                # 实现半监督学习
                if sv_ratio > 0:
                    sv_ratio = max(sv_ratio - 1e-5, 0)
                # 这里并没有进行半监督学习
                # 这里来得到陆地掩膜，不断训练替换
                output_mask = (var_true == 0)
                # -------training for one batch
                # 进行训练
                var_pred = self.mymodel(
                    decoder_input_var.float().to(self.device),
                    train=True,
                    sv_ratio=sv_ratio
                )

                # 计算相关损失
                # loss_var = self.loss_var(var_pred, var_true.float().to(self.device))
                # l2_value = self.L_2_loss(var_pred, var_true.float().to(self.device))
                # loss_var = self.Huber_loss(var_pred, var_true.float().to(self.device))

                loss_var = self.loss_var(var_pred, var_true.float().to(self.device), method)
                var_pred[output_mask] = 0

                # print(loss_var)
                # Loss = loss_var + l2_value

                # 将损失函数进行反向传播
                # 进行权重列表的设置
                # Loss.backward()
                loss_var.backward()
                # 更新参数
                self.opt.step()
                # self.adam.step()
                # 获取得到训练集训练过程中的损失值，获取得到一个当前训练epoch中损失值最小的参数
                if j % 10 == 0:
                    print(
                        "\n-->batch:{} loss_var:{:.2f}".format(
                            j, loss_var
                        )
                    )
                # ---------Intensive verification,强化核查
                # 这段代码的作用是在训练模型的过程中，当训练次数 i_epoch 大于等于 4，且每训练 200 个样本后，获得验证集损失最小的模型参数
                if (i_epoch + 1 >= 1) and (j + 1) % 100 == 0:
                    (_,
                     lossvar_eval,
                     ) = self.model_pred(dataloader=dataloader_eval, method=method)
                    print(
                        "-->Evaluation... \nAVG_loss_var:{:.3f} ".format(
                            lossvar_eval
                        )
                    )
                    # 保存使得模型参数得到最小的损失值，模型参数
                    if lossvar_eval < best:
                        torch.save(
                            self.mymodel, #.state_dict(),
                            chk_path,
                        )
                        best = lossvar_eval
                        count = 0
                        print("\nsaving model...  ")
                        print(best)
            # ----------after one epoch-----------，在进行训练一个epoch之后
            # 获取得到测试eval部分的相关损失值，通过测试集进行比较得到训练损失最小的模型参数。
            (
                _,
                lossvar_eval,
            ) = self.model_pred(dataloader=dataloader_eval, method=method)
            # 打印出相关的损失值
            print(
                "\n-->epoch{} end... \nmin_loss_var:{:.3f}".format(
                    i_epoch, lossvar_eval
                )
            )
            # 如果实际训练损失值小于最小的损失值时，记录count

            if lossvar_eval <= best:
                # 如何大于最小的损失值时，需要保存相关的模型参数，需要进行修改，使得代码能够保存最小损失值的模型参数
                count = 0
                print(
                    "\nsc is decrease from {:.3f} to {:.3f}   \nsaving model...\n".format(
                        best, lossvar_eval
                    )
                )
                # 记录损失最小的模型参数，优化器的状态信息等，需要进行修改
                torch.save(
                    self.mymodel, #.state_dict(),
                    chk_path,
                )
                best = lossvar_eval

            else:
                count += 1
                print("\nsc is not decrease for {} epoch".format(count))

            # ---------early stop,提前达到停止状态
            if mypara.early_stopping:
                if count == self.mypara.patience:
                    print(
                        "\n-----!!!early stopping reached, min(loss)= {:3f}!!!-----".format(
                            best
                        )
                    )
                    break
        if count != 0:
            print(
                "\n-----!!!training stopping , min(loss)= {:3f}!!!-----".format(
                    best
                )
            )
        del self.mymodel


if __name__ == "__main__":

    """
    首先通过配置文件来获取得到相关的数据地址等信息，并输入数据集制作文件SST中，之后将数据集进行切分，得到训练集，验证集。
    测试集：需要在测试文件中单独进行测试
    """
    for method in ['Huber']:
        print(mypara.__dict__)
        
        # 指定数据的起止索引（该模型是训练0时刻数据---》1时刻数据：start_index=0,end_index=1）
        start_index = 0
        end_index = 4
        
        # 根据归一化要求，需要定义一个归一化的类，用于对数据进行归一化
        model_norm_func = StandardScaler()
        error_norm_func = StandardScaler()


        # 制作训练集和测试集
        dataset_train = make_dataset(
            mypara=mypara, start_index=start_index,end_index=end_index,model_norm_func=model_norm_func, error_norm_func=error_norm_func,group=mypara.all_group
        )
        
        
        dataset_val = make_dataset(
            mypara=mypara,start_index=start_index,end_index=end_index,model_norm_func=model_norm_func, error_norm_func=error_norm_func,group=200
        )
        
        # -------------------------------------------------------------
        # 定义一个模型
        trainer = modelTrainer(mypara)
        # 进行模型训练
        trainer.train_model(
            dataset_train=dataset_train,
            dataset_eval=dataset_val,
            method=method
        )
