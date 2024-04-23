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


# 假设 Excel 文件名为 'data.xlsx'，并且数据在第一个 sheet 上
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/data/results/NN/chf_public_NN.csv'

# 读取 Excel 文件
df = pd.read_csv(file_path)

# 移除第一行
df.drop(index=0, inplace=True)
print(df)
# 读取 'CHF' 和 'CHF LUT' 这两列数据
chf_column = df['CHF']
chf_lut_column = df['CHF LUT']
# 将这些列转换为 numpy 数组，并且重新调整形状
chf_column_np = np.array(chf_column, dtype=float).reshape(-1, 1)
chf_lut_column_np = np.array(chf_lut_column, dtype=float).reshape(-1, 1)
print(chf_column_np)
print(chf_lut_column_np)
output_indicator("results", chf_column_np, chf_lut_column_np)