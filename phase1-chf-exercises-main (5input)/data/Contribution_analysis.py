import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建标准化和归一化的scalers
# *****************************************************************************************************************************
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
# scaler_minmax = MinMaxScaler(feature_range=(0, 1))


# Load the CSV file, skipping the first two header lines
# file_path = '/home/user/ZHOU-Wen/phase1-chf-exercises-main/data/inputs/chf_public.csv'
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
normalized_data = data_to_transform.apply(lambda column: scaler_minmax.fit_transform(column.values.reshape(-1, 1)).flatten() if column.name[1] != '-' else column)

# 标准化和归一化的数据将被放置回原始的数据框架中，不包括第一列和第二列
# *****************************************************************************************************************************
data.iloc[:, 2:] = standardized_data
# data.iloc[:, 2:] = normalized_data

# Convert 'Number' and 'Reference ID' columns to numeric
data['Number'] = pd.to_numeric(data['Number'])
data['Reference ID'] = pd.to_numeric(data['Reference ID'])

# Sort the data by 'Number' to maintain the original order
data.sort_values(by='Number', inplace=True)


# Define the feature columns and the label column
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling',
#                    'Inlet Temperature', 'latent_heat_of_vaporization', 'saturated_liquid_enthalpy']
# feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature']
feature_columns = ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality']
label_column = 'CHF'
input_size = len(feature_columns)
output_size = 1

# 选择列
X = data[feature_columns].to_numpy()  # 用列名的列表选择多列
y = data[label_column].to_numpy()

print(X)
print(X.shape)
print(y)
print(y.shape)



from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import pearsonr
import numpy as np

print("---------------------------Filter method---------------------------")
# 过滤方法
# 使用 SelectKBest 选择最好的 K 个特征
# f_classif，这是一种基于ANOVA（方差分析）的测试方法，它适用于连续数据和分类输出。
# 如果想要使用卡方检验方法，可以在SelectKBest中使用chi2作为score_func参数，这样就可以用于选择那些与类别输出最相关的分类输入特征。卡方检验特别适用于那些具有离散分类输出的特征选择任务。
select_k_best = SelectKBest(score_func=f_classif, k='all')
fit = select_k_best.fit(X, y)
scores = fit.scores_
print("ANOVA F-Value scores (Filter method):", scores)
# 这个指标评估了每个特征和目标变量之间的关系强度。值越大，表明特征与输出变量之间的统计显著性越高，因此重要性越大。
# 从结果来看，Heated Length（加热长度）有最高的F值，这意味着它可能是最重要的特征。

# 计算每个特征与目标变量之间的相关系数
correlations = {}
for feature in feature_columns:
    temp_data = data[[feature, label_column]].dropna()  # 删除空值并使用临时变量
    correlation = pearsonr(temp_data[feature], temp_data[label_column])[0]  # 计算相关系数
    correlations[feature] = correlation if not pd.isnull(correlation) else 0

# 打印特征和对应的相关系数
for feature in correlations:
    print("Pearsonr scores (Filter method):",f"{feature}: {correlations[feature]}")
correlation_values = np.array(list(correlations.values()))
print(correlation_values)
# Pearson 相关系数衡量的是线性相关性的强度和方向。负值表示负相关，正值表示正相关。
# Heated Length有最强的负相关（-0.503），而Mass Flux有最强的正相关（0.423）。

# 计算每个特征的互信息得分
mi_scores = mutual_info_regression(X, y)
print("Mutual Information scores (Filter method):", mi_scores)
# 互信息分数衡量了每个特征与目标变量之间的相互依赖性。分数越高，表明特征与目标变量的依赖性越强，因此重要性越大。
# 在这里，Heated Length的分数为1，是最高的，这表示它可能包含最多关于输出变量的信息。

print("---------------------------Wrapper method---------------------------")
# 包装方法
# 使用递归特征消除
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1, step=1)
selector = selector.fit(X, y)
print("Feature ranking (Wrapper method):", selector.ranking_)
# 包装方法通过一系列迭代过程确定特征子集。特征排名告诉我们特征被认为是有序的重要性,这里的值是特征的排名，而不是它们的绝对分数。排名1的特征是最重要的，排名数越大表示重要性越低。
# 根据Feature ranking，Heated Length（加热长度）再次排在第一位，表明它在模型构建中的重要性。

print("---------------------------Embed method---------------------------")
# 嵌入方法
# 使用 LassoCV 和 RidgeCV
lasso = LassoCV(max_iter=10000,).fit(X, y)
ridge = RidgeCV().fit(X, y)
print("Lasso coefficients:", lasso.coef_)
print("Ridge coefficients:", ridge.coef_)
# 这些模型通过正则化来选择特征。非零系数的特征被视为重要的。
# Lasso 回归：非零系数的特征被认为是重要的。系数的绝对值越大，特征的重要性越高。
# Ridge 回归：Ridge回归不会将系数降至零，但是系数的绝对值越大，也表明特征的重要性越高。
# 在Lasso回归中，Inlet Temperature（入口温度）的系数为0，可能表明它是最不重要的特征。
# Ridge回归赋予了所有特征非零权重，但Heated Length（加热长度）的负系数最大，意味着其变化对输出影响最大。

print("---------------------------Model based method---------------------------")
# 使用随机森林
rf = RandomForestRegressor(n_jobs=20)
rf.fit(X, y)
feature_importances = rf.feature_importances_
print("Random Forest feature importances:", feature_importances)
# 随机森林和梯度提升特征重要性：这些值直接表示了特征对模型预测能力的贡献度，值越大表示重要性越高。

# 使用梯度提升树
gb = GradientBoostingRegressor()
gb.fit(X, y)
feature_importances_gb = gb.feature_importances_
print("Gradient Boosting feature importances:", feature_importances_gb)
# 随机森林和梯度提升特征重要性：这些值直接表示了特征对模型预测能力的贡献度，值越大表示重要性越高。

# 排列重要性
perm_importance_result = permutation_importance(rf, X, y, n_repeats=720,n_jobs=20) # 6的阶乘 = 720
perm_importances = perm_importance_result.importances_mean
print("Permutation importances:", perm_importances)
#一个特征的重要性得分越高，意味着这个特征对模型性能的影响越大。

print("---------------------------PCA method---------------------------")
# PCA分析
pca = PCA()
pca.fit(X)
pca_features = pca.components_
print("PCA component variances:", pca.explained_variance_ratio_)
# PCA方法中的各主成分的方差表示了该成分在数据中的信息量。方差值越大，表示那个成分含有越多的信息。第一个主成分占据了最大的方差比例，这意味着它包含了最多的信息。

print("---------------------------Comprehensive evaluation---------------------------")

# Standardize these data（在0和1之间）
# 使用每个方法的最大值来标准化分数
anova_normalized = scores / np.max(scores)
pearson_normalized = np.abs(correlation_values) / np.max(np.abs(correlation_values))
mi_normalized = mi_scores / np.max(mi_scores)

# Convert rankings to scores for consistency with other scores (higher is better)
wrapper_scores = max(selector.ranking_) + 1 - selector.ranking_
wrapper_normalized = wrapper_scores / np.max(wrapper_scores)

lasso_normalized = np.abs(lasso.coef_) / np.max(np.abs(lasso.coef_))
ridge_normalized = np.abs(ridge.coef_) / np.max(np.abs(ridge.coef_))
rf_normalized = feature_importances / np.max(feature_importances)
gb_normalized = feature_importances_gb / np.max(feature_importances_gb)
perm_normalized = perm_importances / np.max(perm_importances)
pca_normalized = pca.explained_variance_ratio_ / np.max(pca.explained_variance_ratio_)

# 汇总标准化分数
total_scores = (anova_normalized + pearson_normalized + mi_normalized + wrapper_normalized +
                lasso_normalized + ridge_normalized + rf_normalized + gb_normalized +
                perm_normalized + pca_normalized)

# 综合特征重要性排序，得分越高表示特征越重要
# 这里使用argsort函数进行排序，它返回从小到大的索引值，所以我们使用[::-1]来翻转顺序得到降序排名
feature_ranking = np.argsort(total_scores)[::-1]

# 输出特征重要性综合得分和排序
print('total_scores:',total_scores)
print('feature_ranking:',feature_ranking)