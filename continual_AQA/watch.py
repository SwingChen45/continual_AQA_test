
import pickle
####FisV标签内容查询
# 文件路径
file_path = 'DATA/FisV/label_5tasks/fisv_tes_train_split_5tasks.pkl'

# 以二进制读取模式打开文件
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看数据类型
print(f"数据类型: {type(data)}")

# 根据图片显示，data[0] 是 ID 列表，data[1] 是分值列表
ids = data[0]
scores = data[1]

print(f"{'样本 ID':<10} | {'对应分值':<10}")
print("-" * 25)

# 使用 zip 将 ID 和 分数 一一对应并打印
for sample_id, score in zip(ids, scores):
    print(f"{sample_id:<10} | {score:<10}")

# 如果你想查看总共有多少条数据
print("-" * 25)
print(f"总数据量: {len(ids)} 条")
# import numpy as np

# # 加载 .npy 文件
# data = np.load('DATA/AQA-7/feat6/train_label.npy')

# # 查看基本信息
# print(f"数据类型: {type(data)}")
# print(f"数组形状: {data.shape}")
# print(f"数据类型: {data.dtype}")

# # 打印具体内容
# print("数据内容：")
# print(data)