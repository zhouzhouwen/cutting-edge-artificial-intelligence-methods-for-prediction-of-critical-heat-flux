import csv

def find_minimum_and_average_value_in_csv(file_name):
    try:
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            min_value = float('inf')
            total = 0
            count = 0
            next(reader)  # 跳过标题行
            for row in reader:
                value = float(row[1])
                if value < min_value:
                    min_value = value
                total += value
                count += 1

            if count > 0:
                average_value = total / count
            else:
                average_value = None

            return min_value, average_value
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# 调用函数并打印最小值和平均值
file_name = '/home/user/ZHOU-Wen/phase1-chf-exercises-main (5input)/Zhou_Wen_Code/Transformer/Transformer_optuna_trials_5.csv'
min_value, average_value = find_minimum_and_average_value_in_csv(file_name)

print("Minimum Value:", min_value)
print("Average Value:", average_value)
