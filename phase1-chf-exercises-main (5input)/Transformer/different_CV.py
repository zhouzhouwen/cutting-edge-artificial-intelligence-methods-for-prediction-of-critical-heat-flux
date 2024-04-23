import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold


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

# 随机生成一个示例 DataFrame
# np.random.seed(0)
# data = pd.DataFrame({
#     'Feature1': np.random.rand(100),  # 第一个特征列
#     'Feature2': np.random.rand(100),  # 第二个特征列
#     'Reference ID': np.random.randint(1, 10, 100)  # 随机生成 Reference ID，假设有9个不同的ID
# })

file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/train_set.csv'
data = pd.read_csv(file_path)

print(data)
group_sample_counts = data['Reference ID'].value_counts()
print(group_sample_counts)
# 应用自定义分割策略
folds = custom__kfold(data)


train_set = data.iloc[folds[0][0]]
test_set = data.iloc[folds[0][1]]

print(train_set)
print(train_set.shape)
print(test_set)
print(test_set.shape)
print('xxxxxxxxxxxxxxxxxxxxx')