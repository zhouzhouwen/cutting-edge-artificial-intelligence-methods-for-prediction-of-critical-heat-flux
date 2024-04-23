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

def split_data(X, y, quantile_bins):
    # 将 y 转换为一维数组
    y = y.ravel()
    # 使用分位数将数据分为多个区间,使用分位数标签来划分数据
    y_binned = pd.qcut(y, q=quantile_bins, labels=False, duplicates='drop')
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)
    y_binned_train_val = pd.qcut(y_train_val, q=quantile_bins, labels=False, duplicates='drop')
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_binned_train_val)

    return X_train, y_train, X_val, y_val, X_test, y_test

# 随机生成一个示例数据矩阵和目标向量
np.random.seed(0)
normalized_data_X = np.random.randint(0, 100, (100, 3))  # 100个样本，每个样本10个特征，整数范围0到99
normalized_data_y = np.random.randint(0, 100, (100, 1))  # 100个样本的目标值，整数范围0到99

# 使用自定义函数分割数据
normalized_data_X_train, y_train, normalized_data_X_val, y_val, normalized_data_X_test, y_test = split_data(normalized_data_X, normalized_data_y, quantile_bins=3)
print(normalized_data_X_train)

a = torch.tensor([1, 2, 3])
print(a)
print(a.shape)
b = a.unsqueeze(0)
print(b)
print(b.shape)

