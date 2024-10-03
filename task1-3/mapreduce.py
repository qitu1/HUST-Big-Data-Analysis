import os
import re
import threading
from collections import defaultdict, Counter
from queue import Queue

# folder路径和words路径
folders = [f'folder_{i}' for i in range(1, 10)]
words_path = 'words.txt'
titles = []

# 读取词汇表
with open(words_path, 'r') as f:
    target_words = set(line.strip().lower() for line in f)
    # 读取词汇表，并将每行转换为小写，去除空白字符后加入集合

# Map阶段函数
def map_task(folder, queue):
    for file in os.listdir(folder):
            titles.append(file[:-4])   # 去掉'.txt'扩展名添加到标题列表中
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as fi:
                content = fi.read().lower()    # 读取文件内容并转换为小写
                words = re.findall(r'\b\w+\b', content)  # 使用正则表达式找到所有单词
                for word in words:
                    if word in target_words:
                        queue.put(((file[:-4], word), 1))   # 将单词和标题作为键，1作为值放入队列

# Reduce阶段函数
def reduce_task(queue, results):
    counter = defaultdict(int)
    while not queue.empty():
        try:
            (key, value) = queue.get_nowait()     # 从队列中获取一个元素
            reduce_results[key] += value  # 更新该键对应的值
        except:
            pass

# 创建队列和线程
queue = Queue()
map_threads = [] # 存储Map阶段的线程
reduce_results = defaultdict(int)

# 启动Map线程
for folder in folders:
    thread = threading.Thread(target=map_task, args=(folder, queue))     # 为每个文件夹创建一个线程
    thread.start()   # 启动线程
    map_threads.append(thread)   # 将线程添加到线程列表中

# 等待Map线程完成
for thread in map_threads:
    thread.join()

# 启动Reduce线程
reduce_threads = []
for _ in range(3):
    thread = threading.Thread(target=reduce_task, args=(queue, reduce_results))
    thread.start()
    reduce_threads.append(thread)

# 等待Reduce线程完成
for thread in reduce_threads:
    thread.join()

reduce_task(queue, reduce_results)
# 统计词频最高的前1000个词汇
word_counts = Counter()   # 创建计数器
for ((title, word), count) in reduce_results.items():
    word_counts[word] += count    # 累加每个单词的计数值

top_1000_words = word_counts.most_common(1000)

# 输出结果
with open('map_results.txt', 'w', encoding='utf-8') as f:
    for key, value in reduce_results.items():
        f.write(f"{key}: {value}\n")

with open('reduce_results.txt', 'w', encoding='utf-8') as f:
    for word, count in top_1000_words:
        f.write(f"{word}: {count}\n")

top_1000_set = set(word for word, _ in top_1000_words)
# 创建跳转关系的字典
jump_relations = defaultdict(set)   # 使用defaultdict创建一个默认值为空集的字典
for ((title, word), count) in reduce_results.items():  #遍历Reduce阶段的结果
    if word in top_1000_set:   # 如果单词在前1000个词汇中
        if title in top_1000_set:   # 如果标题在前1000个词汇中
            if word in titles:   # 如果单词在标题列表中
                jump_relations[title].add(word)   # 将单词添加到标题的跳转关系集中
print(jump_relations)

import pickle
with open("my_jump.pkl","wb") as tf:
    pickle.dump(jump_relations,tf)


with open('jump_relations.txt', 'w', encoding='utf-8') as f:
    for word, targets in jump_relations.items():
        f.write(f"{word} -> {', '.join(targets)}\n")
