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
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from scipy.signal import stft
from torchvision import models
from torchvision.models import VGG13_BN_Weights, ResNet34_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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





def set_seed(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create train and test sets for features and labels from given indices
def create_train_test_sets(data, train_indices, test_indices, feature_cols, label_col):
    X_train = data.iloc[train_indices][feature_cols]
    y_train = data.iloc[train_indices][label_col]
    X_test = data.iloc[test_indices][feature_cols]
    y_test = data.iloc[test_indices][label_col]
    return X_train, y_train, X_test, y_test


# Defining the neural network architecture

class ConvNet(nn.Module):
    def __init__(self, n_layers, channels, kernel_sizes, activation):
        super(ConvNet, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        layers = []
        for i in range(n_layers):
            in_channels = 1 if i == 0 else channels[i - 1]
            out_channels = channels[i]
            kernel_size = kernel_sizes[i]
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
            bn = nn.BatchNorm1d(out_channels)  # 添加批归一化层
            layers.append(nn.Sequential(conv, bn))  # 将卷积层和批归一化层一起添加到层序列中
        self.convs = nn.ModuleList(layers)

        # 计算经过所有卷积层之后的输出长度
        output_length = self.calculate_output_length(5, kernel_sizes)
        self.fc1 = nn.Linear(output_length * out_channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def calculate_output_length(self, input_length, kernel_sizes, ):
        output_length = input_length
        for kernel_size in kernel_sizes:
            output_length = ((output_length + 2 * 1 - kernel_size) // 1) + 1
        return output_length

    def forward(self, x):
        # print(f"Initial shape: {x.shape}")
        if x.ndim == 2:  # 如果数据只有两维，添加通道维度
            x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)  # 使用torch.tanh代替F.tanh
            elif self.activation == 'softsign':
                x = F.softsign(x)
        x = x.view(x.size(0), -1)
        # print(f"After view shape: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"After fc1 shape: {x.shape}")
        x = self.fc2(x)
        # print(f"After fc2 shape: {x.shape}")
        return x



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
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/train_set.csv'
data_train = pd.read_csv(file_path)

# Define the KFold cross-validator
folds = custom__kfold(data_train)

# Define the feature columns and the label column
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1

X_train, y_train, X_val, y_val = create_train_test_sets(data_train, folds[0][0], folds[0][1], feature_columns,label_column)

test_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/test_set.csv'  # 您需要提供实际的文件路径
data_test = pd.read_csv(test_file_path)
X_test = data_test[feature_columns]
y_test = data_test[label_column]



# 为超参数优化选择模型和数据转换类型
n_layers = 3
channels = [16, 64, 32]
kernel_sizes = [4, 2, 3]
activation = 'relu'
batch_size = 128

model = ConvNet(n_layers, channels, kernel_sizes, activation)
# Convert DataFrame/Series to numpy array before converting to Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) if isinstance(X_train,pd.DataFrame) else torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32) if isinstance(X_val,pd.DataFrame) else torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32) if isinstance(X_test,pd.DataFrame) else torch.tensor(X_test, dtype=torch.float32)


y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)






# 加载保存的模型权重
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/model/CNN_best_model_4.pth'))
model.eval()  # 设置为评估模式

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
