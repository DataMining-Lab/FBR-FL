import torch

# 假设你有一个（1，n）的张量
tensor_1xn = torch.tensor([[3, 7, 1, 4, 2]])

# 将其转换为一维张量
tensor_1d = tensor_1xn.view(-1)
print(tensor_1d)
# 获取最大的两个值及其索引
k = 2
top_values, top_indices = torch.topk(tensor_1d, k)
print(top_values.tolist())
print("Top Values:", top_values)
print("Top Indices:", top_indices)
