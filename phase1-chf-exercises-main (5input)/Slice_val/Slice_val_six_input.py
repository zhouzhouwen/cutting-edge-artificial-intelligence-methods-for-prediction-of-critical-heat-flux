import pandas as pd
from sklearn.model_selection import GroupKFold
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

def output_indicator(phase, pre, real):
    print("-------------------------------" + str(phase) + "-------------------------------")

    print("Mean P/M:", "{:.3f}".format(np.average(pre / real)))
    print("Std P/M:", "{:.3f}".format(np.std(pre / real)))
    print("RMSPE:", "{:.3f}".format(np.sqrt(np.mean(np.square((pre - real) / real)))))
    print("MAPE:", "{:.3f}".format(np.mean(np.abs((pre - real) / real))))

    rmse = np.sqrt(np.mean(np.square(real - pre)))
    nrmse_mean = rmse / np.mean(real)
    print("NRMSE:", "{:.3f}".format(nrmse_mean))

    mu = np.mean(real)
    numerator = np.sum((real - pre) ** 2)
    denominator = np.sum((real - mu) ** 2)
    EQ2 = numerator / denominator
    print("Q2:", "{:.3f}".format(EQ2))

slice_real_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/slice_real_value.xlsx'
slice_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/Slice_10.csv'
lut_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/results/LUT/Slice_10_LUT.csv'
sheet_name = 'slice10'
temperature_slice_each = 10
slice_parameter_index = 4
input_features = 6


# ————————————————————————————————————————————————————————————数据处理————————————————————————————————————————————————————————————
data_from_csv = pd.read_csv(slice_file_path)
numeric_data = data_from_csv.iloc[1:].to_numpy(dtype=float)
# Removing the last column which contains NaN values
numeric_data = numeric_data[:, :-1]
# Modifying the 'Pressure' column by dividing by 1e3
numeric_data[:, 2] /= 1e3

# Generating a range of temperatures from 30 to 330 with a step of 30
temperatures = np.arange(30, 351, temperature_slice_each)
# Generating x axis
slice_parameter = numeric_data[:, slice_parameter_index]

# Creating matrix with the new temperature column
slice_data_with_t = [np.column_stack((numeric_data, np.full(numeric_data.shape[0], temp))) for temp in temperatures]
slice_data_2d = np.reshape(slice_data_with_t, (-1, slice_data_with_t[0].shape[1]))

# Load the CSV file, skipping the first two header lines
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)
# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])

# data_to_transform : 从第三列开始的所有数据列
data_to_transform = data.iloc[:, 2:]
data_to_transform = data_to_transform.drop(['Inlet Subcooling', 'CHF'], axis=1)

# 初始化StandardScaler的字典来存储每列的归一化容器
scalers = {}
# 对每列进行归一化
normalized_columns = {}
for column in data_to_transform.columns:
    scaler = StandardScaler()
    normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
    scalers[column] = scaler

# 将归一化后的数据组合成一个新的DataFrame
# normalized_data_to_transform = pd.DataFrame(normalized_columns)
# print(normalized_data_to_transform)
# print(scalers)

# 应用预先定义的scalers对slice_data_2d的每一列进行归一化
normalized_slice_data_2d = np.copy(slice_data_2d)
# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    # print(column_name)
    normalized_slice_data_2d[:, i] = scalers[column_name].transform(slice_data_2d[:, i].reshape(-1, 1)).flatten()


slice_real_data = pd.read_excel(slice_real_file_path, sheet_name=sheet_name, engine='openpyxl')
slice_real_data_values = slice_real_data[['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']].values
for i, column_name in enumerate(scalers):
    # print(column_name)
    slice_real_data_values[:, i] = scalers[column_name].transform(slice_real_data_values[:, i].reshape(-1, 1)).flatten()

# ————————————————————————————————————————————————————————————加载模型————————————————————————————————————————————————————————————
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        position = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        x = x + self.pe[position]
        return x
class TransformerModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.linear_in = nn.Linear(input_features, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 更复杂的输出层
        self.linear_mid = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, 1)

        # 更复杂的输出层
        self.linear_mid1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_mid2 = nn.Linear(d_model, d_model)
        self.linear_mid3 = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.linear_in(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 通过额外的线性层和激活函数
        output = F.softsign(self.linear_mid1(output))
        output = self.dropout(output)
        output = F.softsign(self.linear_mid2(output))
        output = F.softsign(self.linear_mid3(output))
        output = self.linear_out(output)
        return output


d_model = 64
nhead = 32
num_encoder_layers = 4
dim_feedforward = 4096
dropout = 0.01

model = TransformerModel(input_features=input_features, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_fn, output_size, dropout):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0])]
        # layers.append(nn.BatchNorm1d(hidden_layers[0]))  # 添加 BatchNorm 层
        layers.append(activation_fn())  # 激活函数

        # 添加附加层，基于 hidden_layers 列表
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            # layers.append(nn.BatchNorm1d(hidden_layers[i]))  # 添加 BatchNorm 层
            layers.append(activation_fn())  # 添加传递的激活函数

            # 仅在前三层添加 Dropout
            if i < 3:  # 检查层的索引
                layers.append(nn.Dropout(dropout))  # 添加 Dropout 函数

        layers.append(nn.Linear(hidden_layers[-1], output_size))  # 输出层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# hidden_layers = [30, 30, 70, 90, 90, 40, 100]
# activation_fn = nn.ReLU
# criterion = torch.nn.SmoothL1Loss()
# dropout = 0.01
# model = SimpleNN(input_features, hidden_layers, activation_fn, 1, dropout)


# 加载保存的模型权重
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/model/Transformer_best_model_4_6_input.pth'))
model.eval()  # 设置为评估模式

# ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————

input_data = torch.tensor(normalized_slice_data_2d, dtype=torch.float)
input_data = input_data.unsqueeze(0)
with torch.no_grad():
    output = model(input_data)
output = output.cpu().numpy() if output.is_cuda else output.numpy()

input_data_2 = torch.tensor(slice_real_data_values, dtype=torch.float)
input_data_2 = input_data_2.unsqueeze(0)
with torch.no_grad():
    output_2 = model(input_data_2)
output_2 = output_2.cpu().numpy() if output_2.is_cuda else output_2.numpy()

# Extract the StandardScaler or MinMaxScaler
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)
# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])
# print(data)
# data_to_transform : 从第三列开始的所有数据列
data_to_transform = data.iloc[:, 2:]

# 假设scaler_minmax是一个字典，保存了除第一列和第二列外的每列数据的MinMaxScaler实例
scaler_standard = {column: StandardScaler() for column in data_to_transform.columns}

# 假设在适当的地方以列为单位进行了fit_transform操作，比如
for column in scaler_standard:
    # 只针对不含'-'的列进行操作
    if column[1] != '-':
        scaler_standard[column].fit(data_to_transform[[column]])

# 获取最后一列的标签名称
last_column_name = data.columns[-1]

# 为了逆变换，需要确保它是二维的
output = output.reshape(-1, 1)
output_2 = output_2.reshape(-1, 1)
# 使用最后一列的StandardScaler实例进行逆变换
output = scaler_standard[last_column_name].inverse_transform(output)
output_2 = scaler_standard[last_column_name].inverse_transform(output_2)
# ————————————————————————————————————————————————————————————画图————————————————————————————————————————————————————————————
# 将其转换为3维矩阵
output_3d = np.reshape(output, (len(temperatures), 15, 1))
# print(output_3d)
# 使用不同的颜色和标记绘制所有 11 个二维矩阵在一个图上
plt.figure(figsize=(12, 8))

# 为每个图形选择不同的颜色
colors = plt.cm.jet(np.linspace(0, 1, output_3d.shape[0]))

for i in range(output_3d.shape[0]):
    # 横坐标是 Tube Diameter（假设是第一列），纵坐标是 Pre CHF（假设是最后一列）
    plt.plot(slice_parameter, output_3d[i, :, 0], marker='o', color=colors[i], label=f'Temperature {temperatures[i]}℃')


# 读取CSV文件
data_lut = pd.read_csv(lut_file_path)
# 获取最后一列的数据，排除第一个值
last_column_data = data_lut.iloc[:, -1].values[1:].astype(float) / 1000
# 添加LUT数据，假设横坐标与之前的Tube Diameter相同
plt.plot(slice_parameter, last_column_data, marker='x', color='black', label='LUT data', linewidth=2, markersize=8, zorder=1)

plt.scatter(slice_real_data['Outlet Quality'], slice_real_data['CHF'], color='purple', marker='D',  label='Real data', zorder=2)

# Setting title and labels with increased font size
plt.title('Slice parameter vs Pre CHF for different temperature', fontsize=16)
plt.xlabel('Slice parameter [-]', fontsize=14)
plt.ylabel('Pre CHF [kW/m^2]', fontsize=14)

# Enlarging the tick marks' font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

output_indicator("LUT vs Real", slice_real_data['CHF LUT'], slice_real_data['CHF'])
output_indicator("AI vs Real", output_2.flatten(), slice_real_data['CHF'])