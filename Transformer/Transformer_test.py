import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
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



# Defining the neural network architecture
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


def set_seed(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化一个共享的变量来跟踪进度
# progress_bar = tqdm(total=10, desc="Optimizing", unit="trial")

# Function to create train and test sets for features and labels from given indices
def create_train_test_sets(data, train_indices, test_indices, feature_cols, label_col):
    X_train = data.iloc[train_indices][feature_cols]
    y_train = data.iloc[train_indices][label_col]
    X_test = data.iloc[test_indices][feature_cols]
    y_test = data.iloc[test_indices][label_col]
    return X_train, y_train, X_test, y_test

# 重新定义自定义分割策略，以获得整体的训练集和测试集
def custom__kfold(data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    grouped_data = data.groupby('Reference ID')

    # 初始化每个fold的训练集和测试集索引列表
    fold_indices = [[] for _ in range(n_splits)]

    # 对每个组分别应用KFold
    for _, group in grouped_data:
        if len(group) < n_splits:
            # 如果组的样本数小于n_splits，将所有样本放入每个fold的训练集
            for fold_idx in range(n_splits):
                train_indices = group.index
                fold_indices[fold_idx].append((train_indices, np.array([])))
        else:
            # 如果组的样本数足够，正常应用KFold
            for fold_idx, (train_index, test_index) in enumerate(kf.split(group)):
                original_train_index = group.iloc[train_index].index
                original_test_index = group.iloc[test_index].index
                fold_indices[fold_idx].append((original_train_index, original_test_index))

    # 合并每个fold的索引
    final_fold_indices = []
    for fold in fold_indices:
        train_indices = np.concatenate([train_idx for train_idx, _ in fold])
        test_indices = np.concatenate([test_idx for _, test_idx in fold if len(test_idx) > 0])
        final_fold_indices.append((train_indices, test_indices))

    return final_fold_indices

# Load the dataset
train_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/train_set.csv'
data_train = pd.read_csv(train_file_path)

# Define the KFold cross-validator
folds = custom__kfold(data_train)

# Define the feature columns and the label column
feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality', 'latent_heat_of_vaporization', 'saturated_liquid_enthalpy']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1

X_train, y_train, X_val, y_val = create_train_test_sets(data_train, folds[3][0], folds[3][1], feature_columns,label_column)

test_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/test_set.csv'  # 您需要提供实际的文件路径
data_test = pd.read_csv(test_file_path)
X_test = data_test[feature_columns]
y_test = data_test[label_column]


# Model hyperparameters setup
# 超参数的搜索范围
d_model = 64
nhead = 32
num_encoder_layers = 4
dim_feedforward = 4096
dropout = 0.01

model = TransformerModel(input_features=len(feature_columns), d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
# 加载保存的模型权重
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/model/Transformer_best_model_4.pth'))
model.eval()  # 设置为评估模式

X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float).unsqueeze(0)
X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float).unsqueeze(0)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float).unsqueeze(0)



with torch.no_grad():
    y_train_pre = model(X_train_tensor)
    y_val_pre = model(X_val_tensor)
    y_test_pre = model(X_test_tensor)


y_train_pre = y_train_pre.cpu().numpy() if y_train_pre.is_cuda else y_train_pre.numpy()
y_val_pre = y_val_pre.cpu().numpy() if y_val_pre.is_cuda else y_val_pre.numpy()
y_test_pre = y_test_pre.cpu().numpy() if y_test_pre.is_cuda else y_test_pre.numpy()


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
scaler_minmax = {column: MinMaxScaler(feature_range=(-1, 1)) for column in data_to_transform.columns}
# scaler_minmax = {column: MinMaxScaler(feature_range=(0, 1)) for column in data_to_transform.columns}

# 假设在适当的地方以列为单位进行了fit_transform操作，比如
for column in scaler_standard:
    # 只针对不含'-'的列进行操作
    if column[1] != '-':
        scaler_standard[column].fit(data_to_transform[[column]])

# 获取最后一列的标签名称
last_column_name = data.columns[-1]

# 为了逆变换，需要确保它是二维的
y_train_pre = y_train_pre.reshape(-1, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_val_pre = y_val_pre.reshape(-1, 1)
y_val = y_val.to_numpy().reshape(-1, 1)
y_test_pre = y_test_pre.reshape(-1, 1)
y_test = y_test.to_numpy().reshape(-1, 1)


# 使用最后一列的StandardScaler实例进行逆变换
inverse_y_train_pre = scaler_standard[last_column_name].inverse_transform(y_train_pre)
inverse_y_train = scaler_standard[last_column_name].inverse_transform(y_train)
output_indicator("Train", inverse_y_train_pre, inverse_y_train)

inverse_y_val_pre = scaler_standard[last_column_name].inverse_transform(y_val_pre)
inverse_y_val = scaler_standard[last_column_name].inverse_transform(y_val)
output_indicator("Val", inverse_y_val_pre, inverse_y_val)

inverse_y_test_pre = scaler_standard[last_column_name].inverse_transform(y_test_pre)
inverse_y_test = scaler_standard[last_column_name].inverse_transform(y_test)
output_indicator("Test", inverse_y_test_pre, inverse_y_test)

# 使用 concatenate 函数将他们沿着第0轴（行）拼接起来
Total_pre = np.concatenate((inverse_y_train_pre, inverse_y_val_pre, inverse_y_test_pre), axis=0)
Total_real = np.concatenate((inverse_y_train, inverse_y_val, inverse_y_test), axis=0)
output_indicator("Total", Total_pre, Total_real)
