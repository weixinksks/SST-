from myconfig import mypara
import numpy as np
import matplotlib.pylab as plt
import os
from func_for_prediction import func_pre
import torch
import xarray as xr
import matplotlib.colors as colors
from SST import make_dataset
from torch.utils.data import DataLoader, Subset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from Geoformer import Geoformer
# matplotlib.interactive(False)

matplotlib.use('Qt5Agg')

# plt.rc("font", family="Arial")
# mpl.rc("image", cmap="RdYlBu_r")
# plt.rcParams["xtick.direction"] = "in"
# plt.rcParams["ytick.direction"] = "in"
# plt.rcParams["font.family"] = "serif"  # 设置字体家族
# plt.rcParams["font.serif"] = ["Times New Roman"]  # 设置具体字体
# plt.rcParams["font.size"] = 15



# 使用以下的代码来获取训练得到的模型
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L

# 可视化代码-需要考虑删除
def visualize(array, lon, lat,index,type,status):
    """
    array: 2D Numpy array or tensor with the data to be plotted
    lon: tuple with minimum and maximum longitude
    lat: tuple with minimum and maximum latitude
    """

    # 如果输入是PyTorch tensor, 转换为numpy数组
    if torch.is_tensor(array):
        array = array.cpu().numpy()

    # 创建投影
    projection = ccrs.PlateCarree()

    # 创建图和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': projection})

    norm = Normalize(vmin=-8, vmax=8)

    colortable_name = "my_colortable"  # 定义颜色表名称
    cmap_data = np.loadtxt('../Drawing/hotcolr_19lev.rgb')/255.0
    cmap = colors.ListedColormap(cmap_data, colortable_name)
    # cmap = 'coolwarm'  # 'coolwarm' 是一个从蓝色到白色到红色的colormap
    # 绘制数据
    img = ax.imshow(array, extent=lon + lat, transform=projection, origin='upper', cmap=cmap, norm=norm)
    ax.set_title('Data Visualization')
    ax.coastlines()
    # ax.set_ylim(-10, 10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 添加颜色条
    fig.colorbar(img, ax=ax, shrink=0.5)
    # 显示图
    ax.set_title(f"{3*index}hours", fontsize=24)
    save_folder = f"../img/2023.11.13_1/space_{type}_{3*index}_{status}.tif"
    ax.figure.savefig(save_folder, format="TIFF")
    # plt.show()
    print("stop")

# 同样是可视化代码-需要考虑删除

def show_pred_true(all_data, mask_ocean_data,type,status):

    if torch.is_tensor(all_data):
        all_data = all_data.cpu().numpy()


    # 进行图像绘制
    # vmin = np.min(all_data)
    # vmax = np.max(all_data)
    vmin = -2
    vmax = 2


    interval = 0.5
    # 创建自定义的colorbar
    colortable_name = "my_colortable"  # 定义颜色表名称
    cmap_data = np.loadtxt('../Drawing/hotcolr_19lev.rgb') / 255.0
    cmap = colors.ListedColormap(cmap_data, colortable_name)
    for  i in range(all_data.shape[1]):
        # 创建投影
        # projection = ccrs.PlateCarree()
        # 创建图和坐标轴
        # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': projection})
        # 设置画布大小等
        fig, axes = plt.subplots(figsize=(15, 10))

        # 全部数据，包括陆地和海洋数据
        data = all_data[0, i, 0, :, :]
        # data =cut_var_true[0,i,0,:,:]
        # data =cut_var_pred[0,i,0,:,:]-cut_var_true[0,i,0,:,:]
        # # 需要利用掩码信息将陆地信息和海洋信息提取出来
        land_data = np.ma.masked_array(data, ~mask_ocean_data)
        plt.imshow(np.flipud(land_data), cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)

        # 绘制数据图像，并选择自定义的colorbar
        ocean_data = np.ma.masked_array(data, mask_ocean_data)
        plt.imshow(np.flipud(ocean_data), cmap=cmap, vmin=vmin, vmax=vmax)

        # 设置刻度位置

        # lon_ticks = np.arange(file_object_xr['lon'][0], file_object_xr['lon'][-1] + 1, 5)
        # lat_ticks = np.arange(file_object_xr['lat'][0], file_object_xr['lat'][-1] + 1, 5)

        # plt.xticks(np.linspace(0, data.shape[1] - 1, len(lon_ticks)), lon_ticks)
        # plt.yticks(np.linspace(0, data.shape[0] - 1, len(lat_ticks)), np.flipud(lat_ticks))

        plt.colorbar(ticks=np.arange(vmin, vmax + interval, interval))

        # 设置X轴和Y轴刻度标签
        # x_tick_labels = ['105°E', '110°E', '115°E', '120°E', '125°E', '130°E', '135°E', '105°E', '110°E', '115°E', '120°E',
                         # '125°E', '130°E', '135°E', '135°E','140']
        # y_tick_labels = ['40°N', '35°N', '30°N', '25°N', '20°N', '15°N', '40°N', '35°N', '30°N', '25°N']

        # plt.gca().set_xticklabels(x_tick_labels)
        # plt.gca().set_yticklabels(y_tick_labels)

        plt.xlabel('Lon')
        plt.ylabel('Lat')
        plt.suptitle(f"{3 * i}hours", fontsize=24)
        # 显示图形
        plt.show()
        save_folder = f"../img/2023.11.13_1/space_{type}_{3 * i}_{status}.tif"
        plt.savefig(save_folder, format="TIFF")



# 绘制真实误差图像和预测误差图像,进行可视化-需要保留
def spatial_distribution_function(cut_var,mask_ocean_data, type, status):
    # 进行图像绘制
    vmin = -3.0
    vmax = 3.0
    interval = 0.5
    # 创建自定义的colorbar
    colortable_name = "my_colortable"  # 定义颜色表名称
    cmap_data = np.loadtxt('../Drawing/hotcolr_19lev.rgb') / 255.0
    cmap = colors.ListedColormap(cmap_data, colortable_name)
    # index_i = [8, 16, 24]
    # 遍历子图并绘制数据
    for i in range(cut_var.shape[1]):
        fig, axes = plt.subplots(figsize=(15, 10))
        land_data = np.ma.masked_array(cut_var[-1, i, 0, :, :].cpu(), ~mask_ocean_data)
        axes.imshow(np.flipud(land_data), cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)
        # cbar2 = fig.colorbar(im2, ax=axes, ticks=np.arange(vmin, vmax + interval, interval))
        # 绘制数据图像，并选择自定义的colorbar
        ocean_data = np.ma.masked_array(cut_var[-1, i,0,:, :].cpu(), mask_ocean_data)
        im1 = axes.imshow(np.flipud(ocean_data), cmap=cmap, vmin=vmin, vmax=vmax)

        # 绘制旁边的颜色条
        cbar1 = fig.colorbar(im1, ax=axes, ticks=np.arange(vmin, vmax + interval, interval))
        # lon_ticks = np.arange(file_object_xr['lon'][0], file_object_xr['lon'][-1] + 1, 5)
        # lat_ticks = np.arange(file_object_xr['lat'][0], file_object_xr['lat'][-1] + 1, 5)
        #
        # axes.set_xticks(np.linspace(0, cut_var.shape[-1], len(lon_ticks)), lon_ticks)
        # axes.set_yticks(np.linspace(0, cut_var.shape[-2], len(lat_ticks)),
        #                 np.flipud(lat_ticks))
        # # 设置X轴和Y轴刻度标签
        # x_tick_labels = ['30°E', '35°E', '40°E', '45°E', '50°E', '55°E', '60°E', '65°E', '70°E', '75°E', '80°E', '85°E',
        #                  '90°E', '95°E', '100°E']
        # y_tick_labels = ['30°N', '25°N', '20°N', '15°N', '10°N', '5°N', '5°S', '10°S',
        #                  '15°S', '20°S']
        # axes.set_xticklabels(x_tick_labels, fontsize=24, rotation=45)
        # axes.set_yticklabels(y_tick_labels, fontsize=24, rotation=45)



        # 获取颜色条刻度对象
        cbar_ticks = cbar1.ax.get_yticklabels()

        # 设置刻度标签的字体大小
        for tick in cbar_ticks:
            tick.set_fontsize(24)  # 在这里设置字体大小

        # 这里需要判断子图属于那个时间

        axes.set_title(f"{3*i}hours", fontsize=24)
        save_folder = f"../img/2023.11.15_1/space_{type}_{3*i}_{status}.tif"
        axes.figure.savefig(save_folder, format="TIFF")
        # 显示图形
        # plt.show()



# 进行真实误差数据和预测误差数据可视化
def MAE_spatial_distribution_function(cut_var_true, cut_var_pred, index):
    lat_grids_range = mypara.lat_grids_range[1]
    lon_grids_range = mypara.lon_grids_range[1]
    # 这里循环记录每个格点记载修正前后RMSE或者是MAE的空间分布,这里记录了24小时的订正前后的RMSE空间分布。
    MAE_space_matrix = np.zeros((9, lat_grids_range, lon_grids_range))

    # 需要循环计算每个格点的RMSE值，并且需要一个相同规格的二维矩阵来记录计算得到的RMSE值
    for i in range(lat_grids_range):
        for j in range(lon_grids_range):
            # 记录真实的MAE，分别是24时刻,48时刻,72时刻
            MAE_space_matrix[0, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_true[0, 8, 0, i, j])))

            MAE_space_matrix[1, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_true[0, 16, 0, i, j])))
            MAE_space_matrix[2, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_true[0, 24, 0, i, j])))
            # 记录预测的MAE，分别是24时刻,48时刻,72时刻
            MAE_space_matrix[3, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_pred[0, 8, 0, i, j])))
            MAE_space_matrix[4, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_pred[0, 16, 0, i, j])))
            MAE_space_matrix[5, i, j] = torch.mean(torch.abs(torch.tensor(cut_var_pred[0, 24, 0, i, j])))
            # 记录预测-真实的MAE，分别是24时刻,48时刻,72时刻
            MAE_space_matrix[6, i, j] = MAE_space_matrix[0, i, j] - MAE_space_matrix[3, i, j]
            MAE_space_matrix[7, i, j] = MAE_space_matrix[1, i, j] - MAE_space_matrix[4, i, j]
            MAE_space_matrix[8, i, j] = MAE_space_matrix[2, i, j] - MAE_space_matrix[5, i, j]
    # 得到修正前和修正后的RMSE值进行绘制图形
    # 进行图像绘制
    vmin = -3.0
    vmax = 3.0
    interval = 0.5

    # 创建自定义的colorbar
    colortable_name = "my_colortable"  # 定义颜色表名称
    cmap_data = np.loadtxt('../Drawing/hotcolr_19lev.rgb') / 255.0
    cmap = colors.ListedColormap(cmap_data, colortable_name)

    # 遍历子图并绘制数据
    for i in range(9):
        fig, axes = plt.subplots(figsize=(15, 10))
        land_data = np.ma.masked_array(MAE_space_matrix[i, :, :], ~mask_ocean_data)
        axes.imshow(np.flipud(land_data), cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)
        # 绘制数据图像，并选择自定义的colorbar
        ocean_data = np.ma.masked_array(MAE_space_matrix[i, :, :], mask_ocean_data)
        im1 = axes.imshow(np.flipud(ocean_data), cmap=cmap, vmin=vmin, vmax=vmax)

        lon_ticks = np.arange(file_object_xr['lon'][0], file_object_xr['lon'][-1] + 1, 5)
        lat_ticks = np.arange(file_object_xr['lat'][0], file_object_xr['lat'][-1] + 1, 5)

        axes.set_xticks(np.linspace(0, MAE_space_matrix.shape[-1], len(lon_ticks)), lon_ticks)
        axes.set_yticks(np.linspace(0, MAE_space_matrix.shape[-2], len(lat_ticks)),
                        np.flipud(lat_ticks))
        # 设置X轴和Y轴刻度标签
        x_tick_labels = ['30°E', '35°E', '40°E', '45°E', '50°E', '55°E', '60°E', '65°E', '70°E', '75°E', '80°E', '85°E',
                         '90°E', '95°E', '100°E']
        y_tick_labels = ['30°N', '25°N', '20°N', '15°N', '10°N', '5°N', '5°S', '10°S',
                         '15°S', '20°S']

        axes.set_xticklabels(x_tick_labels, fontsize=24, rotation=45)
        axes.set_yticklabels(y_tick_labels, fontsize=24, rotation=45)
        cbar = fig.colorbar(im1, ax=axes, ticks=np.arange(vmin, vmax + interval, interval))

        # 获取颜色条刻度对象
        cbar_ticks = cbar.ax.get_yticklabels()

        # 设置刻度标签的字体大小
        for tick in cbar_ticks:
            tick.set_fontsize(24)  # 在这里设置字体大小
        # 这里需要判断子图属于那个时间
        if i % 3 == 0:
            axes.set_title("24hours", fontsize=24)
        elif i % 3 == 1:
            axes.set_title("48hours", fontsize=24)
        else:
            axes.set_title("72hours", fontsize=24)

        # 在这里将图像进行存储，每一个RMSE空间分布特征子图
        if i < 3:
            save_folder = f"../img/2023.9.20-1/MAE_space_true_{24 * (i % 3 + 1)}_{index}.tif"
            axes.figure.savefig(save_folder, format="TIFF")
        elif 3 <= i < 6:
            save_folder = f"../img/2023.9.20-1/MAE_space_pred_{24 * (i % 3 + 1)}_{index}.tif"
            axes.figure.savefig(save_folder, format="TIFF")
        else:
            save_folder = f"../img/2023.9.20-1/MAE_space_sub_{24 * (i % 3 + 1)}_{index}.tif"
            axes.figure.savefig(save_folder, format="TIFF")
        # 显示图形
    plt.show()

# 显示评价指标的变化情况，这些代码需要保留
def Show_mae_changes(MAE_change, type):
    fig, axes = plt.subplots(figsize=(8, 4))
    x_axis = range(MAE_change.shape[1])
    # 绘制24小时真实误差mae值折线图
    axes.plot(x_axis, MAE_change[8, :], '-o', label='actual 24h', color='blue', markersize=0.5)
    # 绘制48小时真实误差mae值折线图
    axes.plot(x_axis, MAE_change[16, :], '-o', label='actual 48h', color='black', markersize=0.5)
    # 绘制72小时真实误差mae值折线图
    axes.plot(x_axis, MAE_change[24, :], '-o', label='actual 72h', color='red', markersize=0.5)
    axes.set_xlabel('sample numbers')
    axes.set_ylabel('MAE')
    # 这里需要判断是否是mae值还是mae的比例
    if MAE_change[0, 0] < 1:
        axes.set_ylim(0.0, 1)
    else:
        axes.set_ylim(0, 100)

    # 显示网格线
    axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes.legend()
    # 保存图像
    save_folder_24 = f"../img/2023.11.13_1/MAE_err_{type}.tif"
    axes.figure.savefig(save_folder_24, format="TIFF")
    plt.show()




# --------------------------------------------------------
if __name__ == "__main__":
    # 定义一些全局变量，模型的路径，需要去更改模型。
    files = "../model/2023.11.13_1/Geoformer.pkl"
    print(files)

    # 设置输出数据的长度
    lead_max = mypara.output_length

    """# 需要重新进行预测结果的可视化，包括时间纬度和空间纬度上的可视化.
    # 需要在时间纬度上进行可视化制作，即在单点数据上或者是在场数据上,利用时间纬度进行可视化.
    # 需要在空间纬度上进行可视化制作，即需要对于数据在空间维度上进行可视化."""
    # 将模型输入数据地址和模型输出地址，一起传入数据集制作文件SST中

    # 是否可以在func_pre中直接返回关于太平洋模式预报数据的经纬度信息和掩码信息,需要重新更改相关的代码

    adr_data_model = mypara.adr_input


    adr_data_err = mypara.adr_output

    file_object_xr = xr.open_dataset(adr_data_err)
    print(file_object_xr)






    # # 需要获取相关的数据，从而得到对应的经纬度信息和掩码信息
    GLOBAL_LAT = file_object_xr.variables['lat'].values
    GLOBAL_LON = file_object_xr.variables['lon'].values
    mask_ocean_data = file_object_xr["value"][0, 0, :, :].isnull().values


    # 在这里制作测试集，由于进行了归一化需要考虑将数据反归一化，从而正式检验出模型的预测能力和预测精度
    # 1.定义一个归一化方法，作用是对于数据进行归一化,需要保证跟训练集的归一化方法一致。一致采用StandardScaler方法
    model_norm_func = StandardScaler()
    error_norm_func = StandardScaler()

    # 2.利用数据集制作方法，制作数据集，主要采用SST中的制作数据集的方法
    dataset_all = make_dataset(mypara=mypara, model_norm_func=model_norm_func, error_norm_func=error_norm_func)
    print(dataset_all.selectregion())
    # 切分数据集
    train_size = int(mypara.TraindataProportion * len(dataset_all))
    eval_size = len(dataset_all) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset_all, [train_size, eval_size], generator=torch.Generator().manual_seed(0)
    )
    test_group = len(eval_dataset)  # 输出161
    print(test_group)

    # 用来记录相关的指标
    MAE_before = np.empty((25, test_group))
    MAE_after = np.empty((25, test_group))
    MAE_revised = np.empty((25, test_group))

    """
        进行可视化显示，2023-11-13
    """

    # 这里需要进行循环读入，从而实现整个测试集数据的全部可视化
    for index, (cut_var_model,cut_var_pred, cut_var_true) in enumerate(func_pre(
            mypara=mypara,
            adr_model=files,
            eval_dataset=eval_dataset,
            norm_func=error_norm_func
    )):

        # 打印处理的数据形状是[10,25,1,660,780]
        print("模型预测的风数据：",cut_var_pred[0, 0, 0, :, :])
        print("真实的风数据：",cut_var_true[0, 0, 0, :, :])
        """
        重新进行绘制，要求是绘制整体测试样本的mae值。
        细节：
            要求在一张图片上显示24时刻mae,48时刻mae，72时刻mae
            折线图：24时刻是蓝色，48时刻是黑色，72时刻是红色
            存在网格线，
            横坐标：时间
            纵坐标：mae值
        """
        # cut_var_pred  预测结果
        # cut_var_true  真实结果
        # 只需要计算24时刻，48时刻，72时刻上各个样本的MAE值
        for j in range(cut_var_pred.shape[1]):
            # 计算实际误差mae值，
            mae_24_before_revise = torch.mean(torch.abs(torch.tensor( cut_var_true[0, j, 0, :, :].to(mypara.device))))
            # 存储实际误差的mae值
            MAE_before[j, index] = mae_24_before_revise
            # 计算存储叠加预测之后的mae值
            mae_24_after_revise = torch.mean(
                torch.abs(torch.tensor(
                    cut_var_pred[0, j, 0, :, :].to(mypara.device) -  cut_var_true[0, j, 0, :, :].to(mypara.device))))
            # 存储叠加预测的mae值
            MAE_after[j, index] = mae_24_after_revise
            # 计算叠加预测前后的提升比例
            increase_percentage = ((mae_24_before_revise - mae_24_after_revise) / mae_24_before_revise) * 100
            # print(increase_percentage)
            MAE_revised[j, index] = increase_percentage

            # # 在这里显示每一个预测数据和真实数据之间的图像，进行可视化，显示出预测效果
            # show_pred_true(cut_var_pred.cpu(), mask_ocean_data)
            #
            # print(cut_var_true.shape)
            # show_pred_true(cut_var_true.cpu(), mask_ocean_data)

    # np.save('RMSE_before.npy', RMSE_before)
    # np.save('RMSE_after.npy', RMSE_after)





    # 绘制实际误差图像
    Show_mae_changes(MAE_before, 'before')
    # 绘制叠加预测后图像
    Show_mae_changes(MAE_after, 'after')
    # 绘制提升百分比图像
    # print(MAE_revised[0,0])
    Show_mae_changes(MAE_revised, 'revised')

    """
        在这里考虑使用指定下标来读取特定位置的数据
        # 选取预测结果比较好的一个案例
        # 绘制效果提升较好的案例图像
        # # 需要设定不同的current_number来进行索引
    """

    min_indices = np.argmin(MAE_revised, axis=1)
    max_indices = np.argmax(MAE_revised, axis=1)

    print(min_indices)
    print(max_indices)


    # 使用 np.bincount 统计每个索引出现的次数
    min_counts = np.bincount(min_indices)
    max_counts = np.bincount(max_indices)

    # 使用 np.argmax 找到出现次数最多的索引
    most_min_index = np.argmax(min_counts)
    most_max_index = np.argmax(max_counts)

    print(most_min_index)
    print(most_max_index)



    # 绘制一个提升较好的图像
    better_effective_index = [most_max_index]

    subset_better_dataset = Subset(eval_dataset, better_effective_index)

    for index, (cut_var_model,cut_var_pred, cut_var_true) in enumerate(func_pre(
            mypara=mypara,
            adr_model=files,
            eval_dataset=subset_better_dataset,
            norm_func=error_norm_func
    )):


        print(cut_var_pred)
        print(cut_var_true)
        # MAE_spatial_distribution_function(cut_var_true, cut_var_pred, better_effective_index)
        spatial_distribution_function(cut_var_pred,mask_ocean_data,type='Predict', status='better')
        spatial_distribution_function(cut_var_true,mask_ocean_data,type='Error', status='better')
        spatial_distribution_function(cut_var_pred.to(mypara.device)-cut_var_true.to(mypara.device),mask_ocean_data,type='Error-Predict', status='better')


        # show_pred_true(cut_var_pred, mask_ocean_data,type='Predict', status='better')
        # show_pred_true(cut_var_true, mask_ocean_data,type='Error', status='better')
        # show_pred_true(cut_var_pred.to(mypara.device)-cut_var_true.to(mypara.device), mask_ocean_data,type='Error', status='better')
        #

        #
        # for i  in range(mypara.output_length):
        #     # 绘制模式数据
        #     visualize(cut_var_model[0, i, 0, :, :], mypara.lon_range, mypara.lat_range, i, type='model',
        #               status='better')
        #     # 绘制真实误差图像
        #     visualize(cut_var_true[0, i, 0, :, :], mypara.lon_range, mypara.lat_range, i, type='Error',
        #               status='better')
        #     # 绘制预测误差图像
        #     visualize(cut_var_pred[0,i,0,:,:],mypara.lon_range,mypara.lat_range,i,type='Predict',status='better')
        #     visualize(cut_var_true[0, i, 0, :, :].to(mypara.device )-cut_var_pred[0,i,0,:,:].to(mypara.device ),mypara.lon_range,mypara.lat_range,i,type='Error-Predict',status='better')


    # 选取预测结果比较差的一个案例

    bad_effective_index = [most_min_index]
    subset_bad_dataset = Subset(eval_dataset, bad_effective_index)
    # 绘制效果提升较差的案例图像
    for index, (cut_var_model,cut_var_pred, cut_var_true) in enumerate(func_pre(
            mypara=mypara,
            adr_model=files,
            eval_dataset=subset_bad_dataset,
            norm_func=error_norm_func
    )):
        # # MAE_spatial_distribution_function(cut_var_true, cut_var_pred, bed_effective_index)
        spatial_distribution_function(cut_var_pred,mask_ocean_data,type='Predict', status='bad')
        spatial_distribution_function(cut_var_true,mask_ocean_data,type='Error', status='bad')
        spatial_distribution_function(cut_var_pred.to(mypara.device)-cut_var_true.to(mypara.device),mask_ocean_data,type='Error-Predict', status='bad')



        # show_pred_true(cut_var_pred, mask_ocean_data, type='Predict', status='better')
        # show_pred_true(cut_var_true, mask_ocean_data, type='Error', status='better')
        # show_pred_true(cut_var_pred.to(mypara.device)-cut_var_true.to(mypara.device), mask_ocean_data, type='Error',
        #                status='better')

        # for i  in range(mypara.output_length):
        #     # 绘制模式数据
        #     visualize(cut_var_model[0, i, 0, :, :], mypara.lon_range, mypara.lat_range, i, type='model',
        #               status='bad')
        #     # 绘制真实误差图像
        #     visualize(cut_var_true[0, i, 0, :, :], mypara.lon_range, mypara.lat_range, i, type='Error',
        #               status='bad')
        #     # 绘制预测误差图像
        #     visualize(cut_var_pred[0,i,0,:,:],mypara.lon_range,mypara.lat_range,i,type='Predict',status='bad')
        #     visualize(cut_var_true[0, i, 0, :, :].to(mypara.device )-cut_var_pred[0,i,0,:,:].to(mypara.device ),mypara.lon_range,mypara.lat_range,i,type='Error-Predict',status='bad')