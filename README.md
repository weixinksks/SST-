# SST-
A Transformer-based Method for Correcting Daily SST Numerical Forecasting products

项目简介：
本项目提出了一种基于 Transformer 的方法，用于修正每日海表温度（SST）数值预报产品。传统的SST预报方法通常面临模型误差、观测数据不足等问题，本项目采用先进的Transformer结构，结合时空特征，进行数据订正和精度提升。通过本方法，旨在提升数值预报产品的准确性和可靠性，特别是在海洋环境监测、气候预测和导航安全等领域具有广泛的应用前景。

项目结构：



/code                          # 代码部分
|-- /model                      # Transformer模型的实现，包含网络架构、训练过程等
|-- /data_processing            # 数据预处理和增强代码，包含数据加载、标准化等操作
|-- /utils                      # 辅助工具函数，包含日志记录、结果保存等功能
|-- /config                     # 配置文件，包含训练参数、超参数等设置

/data                          # 数据部分
|-- /raw                        # 原始SST数据集，包括ERA5再分析数据等
|-- /processed                  # 处理后的数据集，用于训练和验证
|-- /augmentation               # 数据增强文件，提供数据扩展或扰动方法
