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
feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1


# Defining the neural network architecture
class CNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_fn, output_size):
        super(CNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]  # First layer
        # Add additional layers based on the hidden_layers list
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation_fn())  # Add passed activation function
            layers.append(nn.Dropout(0.01))  # Add Dropout function

        layers.append(nn.Linear(hidden_layers[-1], output_size))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    model.to(device)
    patience = 500
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
                output = model(batch_x)
                val_loss += criterion(output.squeeze(), batch_y).item() * batch_x.size(0)
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
            torch.save(model.state_dict(),
                       '/home/user/ZHOU-Wen/phase1-chf-exercises-main (4input)/Zhou_Wen_Code/model/CNN_best_model_3.pth')
        else:
            patience_counter += 1  # 增加早停计数器
            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # 达到容忍周期，停止训练

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

X_train, y_train, X_val, y_val = create_train_test_sets(data, folds[2][0], folds[2][1], feature_columns,label_column)



# 为超参数优化选择模型和数据转换类型
model_name = 'resnet'
batch_size = 128
image_input_size = 24
transform_type = 'GAF'

# 根据选择的转换类型创建DataLoader
if transform_type == 'GAF':
    X_train = GAF_transform(X_train)
    X_val = GAF_transform(X_val)
else:  # 'MTF'
    X_train = MTF_transform(X_train)
    X_val = MTF_transform(X_val)


# Convert DataFrame/Series to numpy array before converting to Tensor
# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) if isinstance(X_train,pd.DataFrame) else torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32) if isinstance(y_train,pd.Series) else torch.tensor(y_train, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32) if isinstance(X_val, pd.DataFrame) else torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32) if isinstance(y_val, pd.Series) else torch.tensor(y_val, dtype=torch.float32)

X_train_tensor = X_train.repeat(1, 3, 1, 1)  # Repeat the single channel three times to make it three-channel
X_val_tensor = X_val.repeat(1, 3, 1, 1)

# 使用interpolate来调整它们的大小
X_train_tensor = F.interpolate(X_train_tensor, size=(image_input_size, image_input_size), mode='bilinear', align_corners=False)
X_val_tensor = F.interpolate(X_val_tensor, size=(image_input_size, image_input_size), mode='bilinear', align_corners=False)

# Define data loaders for PyTorch
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

criterion = torch.nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) # 使用了初始学习率和L2正则化的值

# Train and evaluate the model
train_losses, val_losses, best_val_loss = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs=1500)

print(best_val_loss)


# 保存训练和验证损失
with open('/home/user/ZHOU-Wen/phase1-chf-exercises-main (4input)/Zhou_Wen_Code/model/CNN_train_losses_3.txt', 'w') as f:
    for loss in train_losses:
        f.write(f"{loss}\n")

with open('/home/user/ZHOU-Wen/phase1-chf-exercises-main (4input)/Zhou_Wen_Code/model/CNN_val_losses_3.txt', 'w') as f:
    for loss in val_losses:
        f.write(f"{loss}\n")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/home/user/ZHOU-Wen/phase1-chf-exercises-main (4input)/Zhou_Wen_Code/model/CNN_loss_plot_3.png')
plt.show()

