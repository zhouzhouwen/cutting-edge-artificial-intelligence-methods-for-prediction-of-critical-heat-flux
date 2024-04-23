import pandas as pd
from sklearn.model_selection import GroupKFold
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

def set_seed(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def output_indicator(phase, pre, real):
    print("-------------------------------"+str(phase)+"-------------------------------")

    print("Mean P/M:", np.average(pre/ real))
    print("Standard Deviation P/M:", np.std(pre / real))
    print("Root Mean Square Percentage Error (RMSPE):",  np.sqrt(np.mean(np.square((pre - real) / real))))
    print("Mean Absolute Percentage Error (MAPE):",  np.mean(np.abs((pre - real) / real)))
    # NRMSE - Normalized by the range of actual values
    rmse = np.sqrt(np.mean(np.square(real - pre)))
    nrmse_mean = rmse / np.mean(real)
    print("Normalized Root Mean Square Error (NRMSE):", nrmse_mean)
    # 计算 mu，即 Y 的平均值
    mu = np.mean(real)
    # 计算分子和分母
    numerator = np.sum((real - pre) ** 2)
    denominator = np.sum((real - mu) ** 2)
    # 计算 EQ^2
    EQ2 = numerator / denominator
    print("Q2 Error:",  EQ2)
    
# ————————————————————————————————————————————————————————————数据处理————————————————————————————————————————————————————————————
# Load the transfer_data file
file_path = '/media/user/wen/CHF_transfer_data/Data_CHF_Zhou_no_filter.csv'
transfer_data = pd.read_csv(file_path)

data_tube = transfer_data.iloc[:1439]
data_annulus = transfer_data.iloc[1439:1817]
data_plate = transfer_data.iloc[1817:]

# Read the specified columns for input and output
input_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality',]
output_column = 'CHF'

# Extract column names for input and output
data_tube_X = data_tube[input_columns].to_numpy()
data_tube_y = data_tube[output_column].to_numpy()

data_annulus_X = data_annulus[input_columns].to_numpy()
data_annulus_y = data_annulus[output_column].to_numpy()

data_plate_X = data_plate[input_columns].to_numpy()
data_plate_y = data_plate[output_column].to_numpy()


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
data_to_transform = data_to_transform.drop(['Inlet Subcooling', 'Inlet Temperature', 'CHF'], axis=1)
# 初始化StandardScaler的字典来存储每列的归一化容器
scalers = {}
# 对每列进行归一化
normalized_columns = {}
for column in data_to_transform.columns:
    scaler = StandardScaler()
    normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
    scalers[column] = scaler


# 应用预先定义的scalers对transfer_data的每一列进行归一化
normalized_data_tube_X = np.copy(data_tube_X)
normalized_data_annulus_X = np.copy(data_annulus_X)
normalized_data_plate_X = np.copy(data_plate_X)

# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    # print(column_name)
    normalized_data_tube_X[:, i] = scalers[column_name].transform(data_tube_X[:, i].reshape(-1, 1)).flatten()
    normalized_data_annulus_X[:, i] = scalers[column_name].transform(data_annulus_X[:, i].reshape(-1, 1)).flatten()
    normalized_data_plate_X[:, i] = scalers[column_name].transform(data_plate_X[:, i].reshape(-1, 1)).flatten()



# ————————————————————————————————————————————————————————————加载模型————————————————————————————————————————————————————————————


# TransformerModel
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
num_encoder_layers = 5
dim_feedforward = 4096
dropout = 0.01

model = TransformerModel(input_features=len(input_columns), d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)


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
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/model/Transformer_best_model_2.pth'))
model.eval()  # 设置为评估模式

# ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————


input_normalized_data_tube_X = torch.tensor(normalized_data_tube_X, dtype=torch.float)
input_normalized_data_annulus_X = torch.tensor(normalized_data_annulus_X, dtype=torch.float)
input_normalized_data_plate_X = torch.tensor(normalized_data_plate_X, dtype=torch.float)

input_normalized_data_tube_X = input_normalized_data_tube_X.unsqueeze(0)
input_normalized_data_annulus_X = input_normalized_data_annulus_X.unsqueeze(0)
input_normalized_data_plate_X = input_normalized_data_plate_X.unsqueeze(0)

with torch.no_grad():
    pre_tube_y = model(input_normalized_data_tube_X)
    pre_annulus_y = model(input_normalized_data_annulus_X)
    pre_plate_y = model(input_normalized_data_plate_X)

pre_tube_y = pre_tube_y.cpu().numpy() if pre_tube_y.is_cuda else pre_tube_y.numpy()
pre_annulus_y = pre_annulus_y.cpu().numpy() if pre_annulus_y.is_cuda else pre_annulus_y.numpy()
pre_plate_y = pre_plate_y.cpu().numpy() if pre_plate_y.is_cuda else pre_plate_y.numpy()


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
pre_tube_y = pre_tube_y.reshape(-1, 1)
pre_annulus_y = pre_annulus_y.reshape(-1, 1)
pre_plate_y = pre_plate_y.reshape(-1, 1)

# 使用最后一列的StandardScaler实例进行逆变换
pre_tube_y = scaler_standard[last_column_name].inverse_transform(pre_tube_y)
pre_annulus_y = scaler_standard[last_column_name].inverse_transform(pre_annulus_y)
pre_plate_y = scaler_standard[last_column_name].inverse_transform(pre_plate_y)

output_indicator("Tube", pre_tube_y, data_tube_y)
output_indicator("Annulus", pre_annulus_y, data_annulus_y)
output_indicator("Plate", pre_plate_y, data_plate_y)