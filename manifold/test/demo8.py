# import random
#
# n = 10  # 要生成的列表长度
# random_list = [random.randint(1, 10) for _ in range(n)]
#
# print("Random list:", random_list)
import random

n = 5  # 你可以修改n的值来生成不同长度的列表
min_value = 1
max_value = 10

# 生成不重复的随机整数列表
random_list = []
while len(random_list) < n:
    random_num = random.randint(min_value, max_value)
    if random_num not in random_list:
        random_list.append(random_num)

print("Random list:", random_list)