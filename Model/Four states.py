import pandas as pd
from sklearn.model_selection import train_test_split

# 定义标签分配函数
def assign_label(row):
    ph = row['pH']
    k_plus = row['K+']
    ca2_plus = row['Ca2+']

    # 根据给定规则分配标签
    if ph <= 7.27 and k_plus >= 5.5:
        return '1'
    elif ph >= 7.42 and k_plus <= 3:
        return '2'
    elif (ca2_plus <= 2.5 and ph >= 7.45) or (ca2_plus <= 2.5 and ph <= 7.30):
        return '3'
    elif 7.35 <= ph <= 7.45 and 3 <= k_plus <= 6 and 2 <= ca2_plus <= 3:
        return '0'

# 假设您的三个文件名为 'file1.xlsx'、'file2.xlsx' 和 'file3.xlsx'
files = ['1.xlsx', '2.xlsx', '3.xlsx']

# 读取并合并所有文件
all_data = pd.concat([pd.read_excel(file) for file in files], ignore_index=True)

# 给数据添加标签
def add_labels_to_data(data):
    data['标签'] = data.apply(assign_label, axis=1)
    return data

# 给合并的数据集添加标签
all_data = add_labels_to_data(all_data)

# 划分 80% 训练集，20% 测试集
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)

# 保存训练集和测试集到 Excel 文件
train_df.to_excel('训练集.xlsx', index=False)
test_df.to_excel('测试集.xlsx', index=False)

print("训练集已保存为 '训练集.xlsx'")
print("测试集已保存为 '测试集.xlsx'")
