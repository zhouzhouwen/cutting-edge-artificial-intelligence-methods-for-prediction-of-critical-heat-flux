import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建标准化和归一化的scalers
# *****************************************************************************************************************************
scaler_standard = StandardScaler()
# scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
# scaler_minmax = MinMaxScaler(feature_range=(0, 1))


# Load the CSV file, skipping the first two header lines
file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public (copy)_with_inlet.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)

# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])
# print(data)

# data_to_transform : 从第三列开始的所有数据列
data_to_transform = data.iloc[:, 2:]
# print(data_to_transform)
# 对数据进行标准化和归一化
# 使用 pandas apply 函数来计算每列
# 注意我们需要使用 lambda 函数来确保每次只传递一列数据给 fit_transform 方法
standardized_data = data_to_transform.apply(lambda column: scaler_standard.fit_transform(column.values.reshape(-1, 1)).flatten() if column.name[1] != '-' else column)
# normalized_data = data_to_transform.apply(lambda column: scaler_minmax.fit_transform(column.values.reshape(-1, 1)).flatten() if column.name[1] != '-' else column)

# 标准化和归一化的数据将被放置回原始的数据框架中，不包括第一列和第二列
# *****************************************************************************************************************************
data.iloc[:, 2:] = standardized_data
# data.iloc[:, 2:] = normalized_data


# Convert 'Number' and 'Reference ID' columns to numeric
data['Number'] = pd.to_numeric(data['Number'])
data['Reference ID'] = pd.to_numeric(data['Reference ID'])

# Sort the data by 'Number' to maintain the original order
data.sort_values(by='Number', inplace=True)

# Define a new sampling function that avoids the beginning and end of each group
'''
total_frac=0.2: 这是整个分组中应该用于测试集的数据点的比例。在这个上下文中，total_frac 被设置为0.2，意味着从每个 Reference ID 分组中抽取总数的20%作为测试数据。
middle_frac=0.8: 这个参数定义了在进行实际抽样之前应该保留用于抽样的分组中间部分的比例。将 middle_frac 设置为0.8意味着避免分组中最初和最后10%的数据点，只从剩下的80%中抽取样本。这是为了避免在测试集中包含可能的极端值或边界情况，而是集中于更典型或更一致的数据点。
'''
def middle_sampling(group, total_frac=0.2, middle_frac=0.8):
    # Define the number of samples to pick as a fraction of the total group size
    total_samples = max(int(round(len(group) * total_frac)), 1)  # Ensure at least 1 sample per group

    # Calculate the start and end indices of the middle portion to avoid extremes if possible
    start_index = int(round(len(group) * (1 - middle_frac) / 2))
    end_index = len(group) - start_index

    # If group is too small and ends overlap, adjust to use the full group, else use the middle
    if start_index >= end_index:
        sampled_indices = range(0, len(group))
    else:
        # Calculate the sample interval within the middle portion
        sample_interval = (end_index - start_index) / total_samples
        sampled_indices = [int(round(start_index + i * sample_interval)) for i in range(total_samples)]

    # Ensure no out of bounds or duplicate indices for small groups
    sampled_indices = list(sorted(set(sampled_indices)))
    sampled_indices = [idx if idx < len(group) else len(group) - 1 for idx in sampled_indices]

    return group.iloc[sampled_indices]
# Apply the even_sampling function to each group to get test set indices
test_set_indices = data.groupby('Reference ID', group_keys=False).apply(middle_sampling).index

# Separate the data into test and training sets
test_set = data.loc[test_set_indices]
train_set = data.drop(test_set_indices)

# Sort the test and training sets by 'Number'
test_set = test_set.sort_values(by='Number')
train_set = train_set.sort_values(by='Number')

# Reset the index for a clean look
test_set.reset_index(drop=True, inplace=True)
train_set.reset_index(drop=True, inplace=True)

# Save the datasets to CSV files
test_set_file = "/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/test_set.csv"
train_set_file = "/home/user/ZHOU-Wen/phase1-chf-exercises-main/Zhou_Wen_Code/dataset/train_set.csv"
test_set.to_csv(test_set_file, index=False)
train_set.to_csv(train_set_file, index=False)
