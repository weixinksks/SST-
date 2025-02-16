import torch
import torch.nn as nn
from my_tools import make_embedding, unfold_func, miniEncoder, miniDecoder, fold_func
from my_tools import compute_land_ratio


# 定义一个transformer模型，来实现一个场到场的预测
class Geoformer(nn.Module):
    # 定义一些模型参数
    def __init__(self, mypara):
        super().__init__()
        # self.mask = mask
        self.mypara = mypara

        d_size = mypara.d_size
        # 定义模型运行路径
        self.device = mypara.device
        print("device", self.device)
        # 判断是否需要风应力，如果需要则在预测模型中额外添加两个数据通道来处理风应力数据
        if self.mypara.needtauxy:
            self.cube_dim_input = (
                    (mypara.input_channal + 2) * mypara.patch_size[0] * mypara.patch_size[1]
            )
            self.cube_dim_output = (
                    (mypara.output_channal + 2) * mypara.patch_size[0] * mypara.patch_size[1]
            )
        else:
            # 不需要风应力数据，则不需要额外的数据通道
            # 计算输入数据的每个“立方体”（cube）的维度大小，
            # 其中一个“立方体”表示输入数据中的一个图像块(patch)其大小为 mypara.patch_size[0] x mypara.patch_size[1]，且输入数据中的每个图像块都包含 mypara.input_channal 个通道
            print(mypara.input_channal)
            print(mypara.output_channal)
            self.cube_dim_input = (
                    mypara.input_channal * mypara.patch_size[0] * mypara.patch_size[1]
            )

            self.cube_dim_output = (
                    mypara.output_channal * mypara.patch_size[0] * mypara.patch_size[1]
            )
            # 进行embedding 操作，创建一个用于将输入数据嵌入到连续向量空间中的嵌入层，其中每个输入的立方体cube都被嵌入为一个大小为d_size 的向量
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim_input,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.input_length,
            device=self.device,
        )
        # 创建一个用于将输出数据嵌入到连续向量空间中的嵌入层，其中每个输入的立方体cube都被嵌入为一个大小为d_size 的向量
        self.predictand_emb = make_embedding(
            cube_dim=self.cube_dim_output,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            max_len=mypara.output_length,
            device=self.device,
        )
        # 创建一层编码部分
        enc_layer = miniEncoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        # 创建一层解码部分
        dec_layer = miniDecoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        # 创建多层编码器
        # self.encoder = multi_enc_layer(
        #     enc_layer=enc_layer, num_layers=mypara.num_encoder_layers
        # )
        # 创建多层解码器
        self.decoder = multi_dec_layer(
            dec_layer=dec_layer, num_layers=mypara.num_decoder_layers
        )

        # 创建线性输出层
        self.linear_output = nn.Linear(d_size, self.cube_dim_output)

        # 创建线性输出层-是否可以创建一个mlp层，经过线性化之后在经过relu等激活函数，从而使得模型预测精度得到提升
        # self.linear_output = nn.Linear(d_size, self.cube_dim_output)
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_size, self.cube_dim_output),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.cube_dim_output, self.cube_dim_output,),
        #     nn.LeakyReLU()
        # )

    def forward(
            self,
            predictand,
            in_mask=None,
            enout_mask=None,
            train=True,
            sv_ratio=0,
    ):
        """
        Args:
            predictor: (batch, lb, C, H, W) 历史数据
            predictand: (batch, pre_len, C, H, W) 未来预测数据 监督样本  真实结果
        Returns:
            outvar_pred: (batch, pre_len, C, H, W) 预测结果
        """

        """
        由于编码器输入和解码器输入是两套数据，最后在预测的时候并不能直接将编码器输入最后一个时刻数据赋值给解码器输入
        现在想法是将编码器输入和解码器输入都置为同一组数据，将真实的数据跟预测数据进行对比计算损失
        10.31:现在由于只能有模式数据去预测误差数据，单纯使用同组数据并不能够满足要求，因此直接将Transformer模型的编码器去除，
        只保留解码器，直接预测。
        """
        # print(predictor.shape)
        # print(predictand.shape)
        # 对于模式数据进行编码并且执行自注意力计算【这里需要去除编码器结构】
        # en_out = self.encode(predictor=predictor, in_mask=in_mask)

        # 进行判断是否模型在进行训练，如果进行训练那么执行一套方法，如果进行评估或者是测试，则执行另一套规则
        if train:
            """
            进行全监督学习。
            """
            """
            2023.11.7_3:现在的想法是首先通过无监督的方法通过模式数据预测误差数据，之后将预测得到的误差数据结合模式数据再次输入Transformer模型中进行训练
            以期望提高Transformer模型的预测能力。
            """

            # 在这里来实现无监督预测数据和模式数据进行整合，实现交叉注意力计算
            # with torch.no_grad():
            #     pred = self.decode(
            #         predictand, in_mask
            #     )

            outvar_pred = self.decode(predictand, in_mask)


        else:
            """
            由于直接利用模式数据来预测误差数据，因此训练过程和验证测试过程一致。此外，直接利用过去的三天数据来预测未来的三天数据，不需要循环预测。
            在训练过程中进行反向传播，梯度计算，但是在验证和测试过程中就不需要计算梯度。
            """
            # with torch.no_grad():
            #     pred = self.decode(
            #         predictand, in_mask
            #     )

            outvar_pred = self.decode(predictand, in_mask)
        return outvar_pred

    # def encode(self, predictor, in_mask):
    #     """
    #     predictor: (B, lb, C, H, W)
    #     en_out: (Batch, S, lb, d_size)
    #     # """
    #     lb = predictor.size(1)
    #     print(lb)
    #
    #     # 判断数据切分的大小
    #     print(self.mypara.patch_size)
    #     print(self.mypara.H0)
    #     print(self.mypara.W0)
    #     print(self.mypara.emb_spatial_size)
    #
    #
    #     predictor = unfold_func(predictor, self.mypara.patch_size, self.mypara.patch_steps)
    #     print(predictor.shape)
    #     predictor = predictor.reshape(predictor.size(0), lb, self.cube_dim_input, -1).permute(
    #         0, 3, 1, 2
    #     )
    #     print(predictor.shape)
    #     # 想法是计算每个patch的陆地比例，从而增加偏置项，以达到调整注意力权重的想法
    #     # 打印出来的数据形状[4, 100, 25, 126]
    #     # 进行单独窗口的数据位置填充
    #     mask_bias = compute_land_ratio(predictor)
    #     # mask_bias = None
    #
    #     predictor = self.predictor_emb(predictor)
    #     # 进入编码层
    #     en_out = self.encoder(predictor, mask_bias.to(self.mypara.device))
    #     # en_out = self.encoder(predictor, mask_bias)
    #     return en_out

    def decode(self, predictand, in_mask=None):
        """
        Args:
            predictand: (B, pre_len, C, H, W)
        output:
            (B, pre_len, C, H, W)
        """
        H, W = predictand.size()[-2:]
        # T : 25
        T = predictand.size(1)

        predictand = unfold_func(predictand, self.mypara.patch_size, self.mypara.patch_steps)
        predictand = predictand.reshape(
            predictand.size(0), T, self.cube_dim_output, -1
        ).permute(0, 3, 1, 2)

        # print(predictand.shape)
        # 对解码器输入进行掩膜偏置计算，计算相关的权重从而调整。
        # mask_bias = compute_land_ratio(predictand)
        predictand = self.predictand_emb(predictand)  # 进行数据embedding处理

        # print(predictand.shape)
        # output = self.decoder(predictand,  mask_bias.to(self.mypara.device))
        output = self.decoder(predictand, in_mask)

        # output = self.mlp(output).permute(0, 2, 3, 1)
        output = self.linear_output(output).permute(0, 2, 3, 1)

        output = output.reshape(
            predictand.size(0),
            T,
            self.cube_dim_output,
            self.mypara.H0,
            self.mypara.W0,
        )
        output = fold_func(
            output, output_size=(H, W), patch_size=self.mypara.patch_size, stride=self.mypara.patch_steps
        )
        # print(output.shape)

        return output

    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.mypara.device)


# 多层的编码部分
# class multi_enc_layer(nn.Module):
#     def __init__(self, enc_layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])
#
#     # 是否可以在这里进行下采样
#     def forward(self, x, mask=None):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return x


# 多层的解码部分
class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, mask_bias):
        # 之后在这里进行上采样
        for layer in self.layers:
            x = layer(x, mask_bias)
        return x
