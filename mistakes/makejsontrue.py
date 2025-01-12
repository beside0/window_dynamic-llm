import json

# 读取原始文件
with open('evaluation_results_parallel_2000.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 将每行数据转换为字典并添加到一个列表中
data = [json.loads(line) for line in lines]

# 将列表转换为 JSON 数组并写入新的文件
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
