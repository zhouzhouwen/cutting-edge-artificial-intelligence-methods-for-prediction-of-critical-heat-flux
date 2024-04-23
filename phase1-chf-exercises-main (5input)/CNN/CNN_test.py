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


def GAF_transform(data):
    data = data.to_numpy()
    # 使用Gramian Angular Field转换一个样本
    gaf = GramianAngularField(image_size=data.shape[1], method='summation')
    data = gaf.fit_transform(data)

    # 画图和颜色条
    # fig, ax = plt.subplots(1, 2, figsize=(18, 6))  # 增加figsize的宽度，为colorbar留出空间
    #
    # # Gramian Angular Field
    # gaf_img = ax[0].imshow(data[0], cmap='rainbow', origin='lower')
    # ax[0].set_title("GAF")
    # fig.colorbar(gaf_img, ax=ax[0], fraction=0.046, pad=0.04)  # 调整colorbar大小和间距
    # #
    # # # Markov Transition Field
    # # mtf_img = ax[1].imshow(X_mtf[0], cmap='rainbow', origin='lower')
    # # ax[1].set_title("MTF")
    # # fig.colorbar(mtf_img, ax=ax[1], fraction=0.046, pad=0.04)
    # #
    # fig.tight_layout()  # 自动调整布局
    #
    # plt.show()

    # 添加单通道层
    # 转换回torch.Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # 添加单通道层
    data_tensor = data_tensor.unsqueeze(1)  # 变为nx1x6x6
    return data_tensor

def MTF_transform(data):
    data = data.to_numpy()
    # 使用Markov Transition Field转换一个样本
    mtf = MarkovTransitionField(image_size=data.shape[1], n_bins=int(data.shape[1]/2))  # 根据需要选择合适的n_bins
    data = mtf.fit_transform(data)

    # 画图和颜色条
    # fig, ax = plt.subplots(1, 2, figsize=(18, 6))  # 增加figsize的宽度，为colorbar留出空间
    #
    #
    # # # Markov Transition Field
    # mtf_img = ax[1].imshow(data[0], cmap='rainbow', origin='lower')
    # ax[1].set_title("MTF")
    # fig.colorbar(mtf_img, ax=ax[1], fraction=0.046, pad=0.04)
    # #
    # fig.tight_layout()  # 自动调整布局
    #
    # plt.show()

    # 添加单通道层
    # 转换回torch.Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # 添加单通道层
    data_tensor = data_tensor.unsqueeze(1)  # 变为nx1x6x6
    return data_tensor

# 修改模型以适用于回归
def modify_model_for_regression(base_model, model_name):
    # 替换分类器以适应回归任务
    if model_name == 'vgg':
        # VGG的分类器是一个序列，包括ReLU层和Dropout层，所以只替换最后一个线性层
        num_features = base_model.classifier[6].in_features  # 获取最后一个线性层的输入特征数
        base_model.classifier[6] = nn.Linear(num_features, 1)  # 替换为一个输出单值的线性层

    elif model_name == 'resnet':
        # ResNet的fc层是单个线性层，直接替换它
        num_features = base_model.fc.in_features  # 获取fc层的输入特征数
        base_model.fc = nn.Linear(num_features, 1)  # 替换为一个输出单值的线性层

    elif model_name == 'efficientnet':
        # 假设模型结构中classifier属性存在且具有线性层
        num_ftrs = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # 保持与原模型相同的dropout
            nn.Linear(num_ftrs, 1)  # 单节点线性层，用于回归输出
        )
    return base_model



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
class CNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_fn, output_size):
        super(CNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]  # First layer
        # Add additional layers based on the hidden_layers list
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation_fn())  # Add passed activation function
            layers.append(nn.Dropout(0.1))  # Add Dropout function

        layers.append(nn.Linear(hidden_layers[-1], output_size))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



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
feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1

X_train, y_train, X_val, y_val = create_train_test_sets(data_train, folds[0][0], folds[0][1], feature_columns,label_column)

test_file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/test_set.csv'  # 您需要提供实际的文件路径
data_test = pd.read_csv(test_file_path)
X_test = data_test[feature_columns]
y_test = data_test[label_column]



# 为超参数优化选择模型和数据转换类型
model_name = 'resnet'
batch_size = 128
image_input_size = 24
transform_type = 'GAF'

# 根据选择的转换类型创建DataLoader
if transform_type == 'GAF':
    X_train = GAF_transform(X_train)
    X_val = GAF_transform(X_val)
    X_test = GAF_transform(X_test)
else:  # 'MTF'
    X_train = MTF_transform(X_train)
    X_val = MTF_transform(X_val)
    X_test = MTF_transform(X_test)

# Convert DataFrame/Series to numpy array before converting to Tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)

X_train_tensor = X_train.repeat(1, 3, 1, 1)  # Repeat the single channel three times to make it three-channel
X_val_tensor = X_val.repeat(1, 3, 1, 1)
X_test_tensor = X_test.repeat(1, 3, 1, 1)

# 使用interpolate来调整它们的大小
X_train_tensor = F.interpolate(X_train_tensor, size=(image_input_size, image_input_size), mode='bilinear', align_corners=False)
X_val_tensor = F.interpolate(X_val_tensor, size=(image_input_size, image_input_size), mode='bilinear', align_corners=False)
X_test_tensor = F.interpolate(X_test_tensor, size=(image_input_size, image_input_size), mode='bilinear', align_corners=False)


# 初始化模型
if model_name == 'vgg':
    weights = VGG13_BN_Weights.DEFAULT  # Or another weight enum as needed
    model = models.vgg13_bn(weights=weights)
    model = modify_model_for_regression(model, 'vgg')
elif model_name == 'efficientnet':
    weights = EfficientNet_B1_Weights.DEFAULT  # Or another weight enum as needed
    model = models.efficientnet_b1(weights=weights)
    model = modify_model_for_regression(model, 'efficientnet')
elif model_name == 'resnet':
    weights = ResNet34_Weights.DEFAULT  # Or another weight enum as needed
    model = models.resnet34(weights=weights)
    model = modify_model_for_regression(model, 'resnet')


# 加载保存的模型权重
model.load_state_dict(torch.load('/home/user/ZHOU-Wen/phase1-chf-exercises-main (4input)/Zhou_Wen_Code/model/CNN_best_model_3.pth'))
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
