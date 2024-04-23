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
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
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
num_encoder_layers = 5
dim_feedforward = 4096
dropout = 0.01

# 将模型移到GPU上
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_features=len(feature_columns), d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout).to(device)

# 加载保存的模型权重
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/model/Transformer_best_model_2.pth'))
model.eval()  # 设置为评估模式

# 将数据移到GPU上
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float).unsqueeze(0).to(device)
X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float).unsqueeze(0).to(device)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float).unsqueeze(0).to(device)

# 使用GPU进行预测
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
print(data)
# data_to_transform : 从第三列开始的所有数据列
data_to_transform = data.iloc[:, 2:]
print(data_to_transform)
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


columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
all_input = data_to_transform[columns]
for column in columns:
    all_input[column] = scaler_standard[column].inverse_transform(all_input[[column]])

all_input = torch.tensor(all_input.to_numpy(), dtype=torch.float).unsqueeze(0).to(device)

# 使用GPU进行预测
with torch.no_grad():
    all_input_pre = model(all_input)

all_input_pre = all_input_pre.cpu().numpy() if all_input_pre.is_cuda else all_input_pre.numpy()

if isinstance(all_input, torch.Tensor):
    all_input = all_input.numpy()  # 转换为 numpy 数组
    all_input = all_input.squeeze()

all_input_pre = np.squeeze(all_input_pre)
all_input_df = pd.DataFrame(all_input)
all_input_pre_df = pd.DataFrame(all_input_pre)
# Concatenate all dataframes horizontally
final_df = pd.concat([all_input_df.reset_index(drop=True), all_input_pre_df], axis=1)

# Save the combined dataframe to a CSV file
final_df.to_csv('Final_Dataset.csv', index=False)






plt.figure(figsize=(8, 8))
plt.scatter(Total_real, Total_pre, alpha=0.5)
# Line of perfect agreement
line = np.linspace(0, np.max([Total_real.max(), Total_pre.max()]), 100)
plt.plot(line, line, 'b-', label='Perfect Agreement', lw=2)
# Lines for +/- 10% of the perfect agreement
plt.plot(line, line * 1.1, 'r--', label='+10% error')
plt.plot(line, line * 0.9, 'r--', label='-10% error')
plt.axis('equal')  # 设置坐标轴的比例为等比例
# Labeling the axes
plt.xlabel('Measured CHF [kW/m^2]')
plt.ylabel('Predicted CHF [kW/m^2]')
# Adding legend
plt.legend()
# Show plot
plt.grid()
plt.show()


errors = (Total_pre - Total_real) / Total_real
# 绘制直方图
plt.figure(figsize=(10, 6))
counts, bins, bars = plt.hist(errors, bins=np.arange(-1, 1.1, 0.05), color='blue', edgecolor='black', alpha=0.5)

# 在误差为零的位置绘制一条垂直的黑色实线
plt.axvline(x=0, color='blue', linestyle='-', linewidth=2)

# 绘制+/-10%的误差带
plt.axvline(x=-0.1, color='red', linestyle='--', label='±10% error')
plt.axvline(x=0.1, color='red', linestyle='--')

xticks = np.arange(-1, 1.1, 0.1)
xtick_labels = [f'{int(round(tick*100))}%' for tick in xticks]
plt.xticks(xticks, xtick_labels)


yticks = plt.yticks()[0]
print(yticks)
plt.yticks(yticks, [f'{int(round(ytick/24579*100))}%' for ytick in yticks])

# 添加图例
plt.legend()
# 添加坐标轴标签
plt.xlabel('Error bands (%)')
plt.ylabel('% of Data')
plt.grid()
# 显示图表
plt.show()

# Load the CSV file, skipping the first two header lines
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public.csv'
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
data = pd.read_csv(file_path, usecols=feature_columns)
data_to_transform = data.iloc[1:, :]
# 重置索引
data_to_transform = data_to_transform.reset_index(drop=True)
# 初始化标准化器
scaler = StandardScaler()
# 拟合和转换数据
data_normalized = scaler.fit_transform(data_to_transform)
data_normalized = torch.tensor(data_normalized, dtype=torch.float).unsqueeze(0)

with torch.no_grad():
    total_pre_in_order = model(data_normalized)
total_pre_in_order = total_pre_in_order.cpu().numpy() if total_pre_in_order.is_cuda else total_pre_in_order.numpy()

total_pre_in_order = total_pre_in_order.reshape(-1, 1)
total_pre_in_order = scaler_standard[last_column_name].inverse_transform(total_pre_in_order)
total_real_in_order = pd.read_csv(file_path, usecols=['CHF']).iloc[1:, :].astype(float)
total_real_in_order = total_real_in_order.values
output_indicator("total_pre_in_order", total_pre_in_order, total_real_in_order)

total_errors = (total_pre_in_order-total_real_in_order)/total_real_in_order


fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Column names for usecols
read_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality',]
plot_columns = ['Tube Diameter [m]', 'Heated Length [m]',  'Pressure [kPa]', 'Mass Flux [kg/m^2/s]', 'Outlet Quality [-]', ]

# Plotting on each subplot with specific columns from the CSV
for i, (read_col, plot_col) in enumerate(zip(read_columns, plot_columns)):
    data = pd.read_csv(file_path, usecols=[read_col]).iloc[1:, :].astype(float).values
    ax = axs[i // 2, i % 2]
    ax.scatter(data, total_errors, alpha=0.5)
    ax.axhline(0, color='blue', lw=2)
    ax.axhline(1.0, color='red', ls='--')
    ax.axhline(-1.0, color='red', ls='--')
    ax.set_xlabel(plot_col)
    ax.set_ylabel('Relative Error [%]')
    ax.set_ylim(-2.0, 4.0)
    ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])  # Set the positions for the ticks
    ax.grid()
    ax.set_yticklabels(['-200%', '-100%', '0%', '100%', '200%', '300%', '400%'])  # Set the labels for the ticks


plt.tight_layout()
plt.show()
