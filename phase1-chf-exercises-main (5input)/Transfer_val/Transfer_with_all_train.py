import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

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
    print("-------------------------------" + str(phase) + "-------------------------------")

    print("Mean P/M:", np.average(pre / real))
    print("Standard Deviation P/M:", np.std(pre / real))
    print("Root Mean Square Percentage Error (RMSPE):", np.sqrt(np.mean(np.square((pre - real) / real))))
    print("Mean Absolute Percentage Error (MAPE):", np.mean(np.abs((pre - real) / real)))
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
    print("Q2 Error:", EQ2)

# ————————————————————————————————————————————————————————————数据处理————————————————————————————————————————————————————————————
# Load the transfer_data file
file_path = '/media/user/wen/CHF_transfer_data/Data_CHF_Zhou_no_filter.csv'
transfer_data = pd.read_csv(file_path)

data_tube = transfer_data.iloc[:1439]
data_annulus = transfer_data.iloc[1439:1817]
data_plate = transfer_data.iloc[1817:]

# file_path = '/media/user/wen/CHF_transfer_data/Data_CHF_Zhou.csv'
# transfer_data = pd.read_csv(file_path)
#
# data_tube = transfer_data.iloc[:1137]
# data_annulus = transfer_data.iloc[1137:]

# Read the specified columns for input and output
input_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality',]
output_column = 'CHF'

# Extract column names for input and output
data_X = data_plate[input_columns].to_numpy()
data_y = data_plate[output_column].to_numpy()

base_model_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/model/Transformer_best_model_2.pth'
new_model_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/Transfer_val/Transfer_model/Transfer_with_all_train_tube.pth'

# Load the CSV file, skipping the first two header lines
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)
# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])


# 这里是为了标准化输入
data_to_transform = data.iloc[:, 2:]
data_to_transform = data_to_transform.drop(['Inlet Subcooling', 'Inlet Temperature', 'CHF'], axis=1)
# 初始化StandardScaler的字典来存储每列的归一化容器
scalers = {}
# 对每列进行标准化
normalized_columns = {}
for column in data_to_transform.columns:
    scaler = StandardScaler()
    normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
    scalers[column] = scaler

# 应用预先定义的scalers对transfer_data的每一列进行归一化
normalized_data_X = np.copy(data_X)

# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    # print(column_name)
    normalized_data_X[:, i] = scalers[column_name].transform(data_X[:, i].reshape(-1, 1)).flatten()


# 这里是为了标准化输出
data_to_transform = data.iloc[:, 2:]
data_to_transform = data_to_transform.drop(['Inlet Subcooling', 'Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature'], axis=1)
# 初始化StandardScaler的字典来存储每列的归一化容器
scalers = {}
# 对每列进行标准化
normalized_columns = {}
for column in data_to_transform.columns:
    scaler = StandardScaler()
    normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
    scalers[column] = scaler

# 应用预先定义的scalers对transfer_data的每一列进行归一化
data_y = data_y.astype(float) # 非常非常重要，如果不进行浮点数，产生的标准化的结果居然整数！！！！
data_y = data_y.reshape(-1, 1)
normalized_data_y = np.copy(data_y)

# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    # print(column_name)
    normalized_data_y[:, i] = scalers[column_name].transform(data_y[:, i].reshape(-1, 1)).flatten()

def split_data(X, y, quantile_bins):
    # 将 y 转换为一维数组
    y = y.ravel()
    # 使用分位数将数据分为多个区间,使用分位数标签来划分数据
    y_binned = pd.qcut(y, q=quantile_bins, labels=False, duplicates='drop')
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)
    y_binned_train_val = pd.qcut(y_train_val, q=quantile_bins, labels=False, duplicates='drop')
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_binned_train_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, y_train, X_val, y_val, X_test, y_test

normalized_data_X_train, y_train, normalized_data_X_val, y_val, normalized_data_X_test, y_test = split_data(normalized_data_X, normalized_data_y, quantile_bins=3)


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
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, batch_first=True)
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

batch_size = 32
d_model = 64
nhead = 32
num_encoder_layers = 5
dim_feedforward = 4096
dropout = 0.01

model = TransformerModel(input_features=len(input_columns), d_model=d_model, nhead=nhead,
                         num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)

# 加载保存的模型权重
model.load_state_dict(torch.load(base_model_path))

# ————————————————————————————————————————————————————————————再次训练————————————————————————————————————————————————————————————
# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(normalized_data_X_train, y_train)
val_dataset = TensorDataset(normalized_data_X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 定义您的模型、优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = torch.nn.SmoothL1Loss()

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    model.to(device)
    patience = 5000
    patience_counter = 0  # 初始化早停计数器
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=150,verbose=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            # for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度

            optimizer.zero_grad()
            output = model(batch_x)

            loss = criterion(output.squeeze(), batch_y.squeeze())
            train_loss += loss.item() * batch_x.size(0)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                batch_x = batch_x.unsqueeze(0)  # 适合任务的序列长度

                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y.squeeze()).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)


        total_loss = train_loss + val_loss
        # 更新学习率
        scheduler.step(total_loss)
        # Early Stopping check
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patience_counter = 0  # 重置早停计数器
            # Optional: Save best model
            print('Best')
            torch.save(model.state_dict(), new_model_path)
        else:
            patience_counter += 1  # 增加早停计数器
            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # 达到容忍周期，停止训练

        # 打印损失
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}')

    return train_losses, val_losses, best_val_loss

# Train and evaluate the model
train_losses, val_losses, best_val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=2000)


# ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————


model = TransformerModel(input_features=len(input_columns), d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
# 加载保存的模型权重
model.load_state_dict(torch.load(new_model_path))
model.eval()  # 设置为评估模式

normalized_data_X_train = normalized_data_X_train.unsqueeze(0)
normalized_data_X_val = normalized_data_X_val.unsqueeze(0)
normalized_data_X_test = normalized_data_X_test.unsqueeze(0)

with torch.no_grad():
    y_train_pre = model(normalized_data_X_train)
    y_val_pre = model(normalized_data_X_val)
    y_test_pre = model(normalized_data_X_test)

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
# scaler_minmax = {column: MinMaxScaler(feature_range=(-1, 1)) for column in data_to_transform.columns}
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
y_train = y_train.numpy().reshape(-1, 1)
y_val_pre = y_val_pre.reshape(-1, 1)
y_val = y_val.numpy().reshape(-1, 1)
y_test_pre = y_test_pre.reshape(-1, 1)
y_test = y_test.numpy().reshape(-1, 1)


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

print(Total_pre)
print(Total_real)
print('xxxxxxxxxxxxxxx')