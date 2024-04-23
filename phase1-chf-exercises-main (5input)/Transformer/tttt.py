
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
import math
import matplotlib.pyplot as plt

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

# 重新定义自定义分割策略，以获得整体的训练集和测试集
def custom__kfold(data, n_splits=100):
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
data = pd.read_csv(file_path)

# Define the KFold cross-validator
folds = custom__kfold(data)

# Define the feature columns and the label column
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Temperature']
# feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1



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

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 使用Xavier初始化方法初始化线性层的参数
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.constant_(self.linear_in.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_mid1.weight)
        nn.init.constant_(self.linear_mid1.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_mid2.weight)
        nn.init.constant_(self.linear_mid2.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_mid3.weight)
        nn.init.constant_(self.linear_mid3.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0.0)

    def forward(self, src):
        src = self.linear_in(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 通过额外的线性层和激活函数
        output = F.softsign(self.linear_mid1(output))
        # output = self.dropout(output)
        output = F.softsign(self.linear_mid2(output))
        output = F.softsign(self.linear_mid3(output))
        output = self.linear_out(output)
        return output

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    model.to(device)
    patience = 1000
    patience_counter = 0  # 初始化早停计数器
    train_losses = []
    val_losses = []
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=150,verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.99)

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
            loss = criterion(output.squeeze(), batch_y)
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
                val_loss += criterion(output.squeeze(), batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)


        total_loss = train_loss + val_loss
        # 更新学习率
        scheduler.step()
        # Early Stopping check
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patience_counter = 0  # 重置早停计数器
            # Optional: Save best model
            torch.save(model.state_dict(), '/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/model/Transformer_best_model_22222.pth')
            print(f'Best Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}')
        # else:
        #     patience_counter += 1  # 增加早停计数器
        #     if patience_counter >= patience:
        #         print("Early stopping triggered")
        #         break  # 达到容忍周期，停止训练

        # 打印损失
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}')

    return train_losses, val_losses, best_val_loss



set_seed(42)
# Create train and test sets

# X_train_fold_1, y_train_fold_1, X_val_fold_1, y_val_fold_1 = create_train_test_sets(data, folds[0][0], folds[0][1], feature_columns, label_column)
# X_train_fold_2, y_train_fold_2, X_val_fold_2, y_val_fold_2 = create_train_test_sets(data, folds[1][0], folds[1][1], feature_columns, label_column)
# X_train_fold_3, y_train_fold_3, X_val_fold_3, y_val_fold_3 = create_train_test_sets(data, folds[2][0], folds[2][1], feature_columns, label_column)
# X_train_fold_4, y_train_fold_4, X_val_fold_4, y_val_fold_4 = create_train_test_sets(data, folds[3][0], folds[3][1], feature_columns, label_column)
# X_train_fold_5, y_train_fold_5, X_val_fold_5, y_val_fold_5 = create_train_test_sets(data, folds[4][0], folds[4][1], feature_columns, label_column)

X_train, y_train, X_val, y_val = create_train_test_sets(data, folds[4][0], folds[4][1], feature_columns,label_column)

# Convert DataFrame/Series to numpy array before converting to Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) if isinstance(X_train,pd.DataFrame) else torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32) if isinstance(X_val, pd.DataFrame) else torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)

# Define data loaders for PyTorch
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model hyperparameters setup
# 超参数的搜索范围
d_model = 64
nhead = 32
num_encoder_layers = 5
dim_feedforward = 4096
dropout = 0.01

# criterion = torch.nn.SmoothL1Loss()
criterion = torch.nn.L1Loss()
model = TransformerModel(input_features=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# Train and evaluate the model
train_losses, val_losses, best_val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=60000)

print(best_val_loss)


