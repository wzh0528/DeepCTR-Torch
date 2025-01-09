import pandas as pd
import numpy as np

# 创建一个示例 DataFrame
data = {
    'lists': [[11, 13], [0], [10, 5, 6, 7], [8, 9]]  # 列表长度不一致
}
df = pd.DataFrame(data)

def pad_and_increment(lst, max_len):
    # Increment each element by 1
    incremented = [x + 1 for x in lst]
    # Pad with 0 to make the list of max_len length
    incremented.extend([0] * (max_len - len(incremented)))
    return incremented

# 求最大列表长度
max_length = df['lists'].apply(len).max()

# 对每个列表元素进行操作（加1，并填充至最大长度），并转换为np.ndarray对象
padded_lists = df['lists'].apply(lambda lst: pad_and_increment(lst, max_length)).tolist()

# 转换为NumPy数组
padded_arrays = np.array(padded_lists)


print(df['lists'])

# 添加到DataFrame或进行其他操作
df['padded_lists'] = list(padded_arrays)  # 这里的list()调用确保每行是NumPy数组对象

print("DataFrame after processing:")
print(df)

print("\nThe resulting NumPy array:")
print(padded_arrays)