import pandas as pd

# 读取 Excel 文件
with pd.ExcelFile('/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/Transformer/Final_Dataset.xlsx') as xls:
    df1 = pd.read_excel(xls, 'Sheet1')  # 读取第一个sheet
    df2 = pd.read_excel(xls, 'Sheet2')  # 读取第二个sheet

# 指定要匹配的列
columns_to_match = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'CHF']

# 使用 merge 方法合并两个 DataFrame
merged_df = pd.merge(df1, df2[['Number', 'Reference ID'] + columns_to_match], on=columns_to_match, how='left')

# 将合并后的 DataFrame 写回 Excel
with pd.ExcelWriter('/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/Transformer/Final_Dataset.xlsx', engine='openpyxl', mode='a') as writer:
    writer.book.remove(writer.book['Sheet1'])  # 删除原来的 Sheet1
    merged_df.to_excel(writer, sheet_name='Sheet1', index=False)  # 写入新的 Sheet1
