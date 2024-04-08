import numpy as np

def top_n_indices(tensor, n):
    # 找到最大的n个值的索引
    indices = np.argsort(tensor)[-n:][::-1]
    return indices

# 示例输入一维张量
tensor = np.array([10, 5, 8, 3, 15, 20, 12, 6])
n = 3  # 要找到的最大值的数量

top_indices = top_n_indices(tensor, n)
print("Top", n, "indices:", top_indices)
