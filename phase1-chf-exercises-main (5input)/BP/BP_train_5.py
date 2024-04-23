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


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    model.to(device)
    patience = 500
    patience_counter = 0  # 初始化早停计数器

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200,verbose=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False):
            # for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            train_loss += loss.item() * batch_x.size(0)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)


        total_loss = train_loss + val_loss
        # 更新学习率
        scheduler.step(total_loss)
        # Early Stopping check
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patience_counter = 0  # 重置早停计数器
            # Optional: Save best model
            # torch.save(model.state_dict(), '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/model/BP_best_model_1.pth')
        else:
            patience_counter += 1  # 增加早停计数器
            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # 达到容忍周期，停止训练
    return best_val_loss


def objective(trial):

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

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model hyperparameters setup
    n_layers = trial.suggest_int('n_layers', 4, 13)
    hidden_layers = [trial.suggest_int('n_units_l{}'.format(i), 30, 100, step=10) for i in range(n_layers)]
    dropout = trial.suggest_categorical('dropout', [0.01, 0.03, 0.05])
    # initial_lr = trial.suggest_categorical('initial_lr', [0.1, 0.01, 0.001])

    activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'softsign'])
    # 根据获取的名称，使用正确的PyTorch激活函数
    if activation_name == 'relu':
        activation_fn = nn.ReLU
    elif activation_name == 'tanh':
        activation_fn = nn.Tanh
    elif activation_name == 'softsign':
        activation_fn = nn.Softsign

    # loss_name = trial.suggest_categorical('loss', ['MSE', 'MAE', 'Huber'])
    # # 根据获取的名称，使用正确的PyTorch损失函数
    # if loss_name == 'MSE':
    #     criterion = torch.nn.MSELoss()
    # elif loss_name == 'MAE':
    #     criterion = torch.nn.L1Loss()
    # elif loss_name == 'Huber':
    #     criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.SmoothL1Loss()

    model = SimpleNN(input_size, hidden_layers, activation_fn, output_size, dropout)  # Assuming input_size and output_size are known
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) # 使用了初始学习率和L2正则化的值

    # Train and evaluate the model
    val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=3000)

    return val_loss


# 定义一个用于保存每次试验数据的回调函数
def save_trial_callback(study, trial):
    trial_record = {
        "number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "datetime_start": trial.datetime_start,
        "datetime_complete": trial.datetime_complete
    }
    # 将每次试验的记录追加到CSV文件中
    pd.DataFrame([trial_record]).to_csv("BP_optuna_trials_5.csv", mode='a', header=False, index=False)

# 在开始优化之前重置CSV文件
columns = ["number", "value", "params", "datetime_start", "datetime_complete"]
pd.DataFrame(columns=columns).to_csv("BP_optuna_trials_5.csv", index=False)

# Running the Optuna optimization
sampler = optuna.samplers.NSGAIISampler()
study = optuna.create_study(sampler=sampler, direction='minimize')
study.optimize(objective, n_trials=200, callbacks=[save_trial_callback], n_jobs=1)



# Fetching the best model parameters
best_trial = study.best_trial
print(best_trial)

