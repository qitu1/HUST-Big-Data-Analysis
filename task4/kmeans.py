import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# KMeans 聚类函数
def kmeans(data, n_clusters=3, max_iter=300, tol=1e-6, random_state=None):
    # 初始化随机种子
    random_state = np.random.RandomState(random_state)
    # 随机选择初始质心
    indices = random_state.choice(data.shape[0], n_clusters, replace=False)
    centroids = data[indices]
    
    # 初始化变量
    labels = None
    for i in range(max_iter):
        # 计算距离
        distances = np.sqrt((data - centroids[:, np.newaxis])**2).sum(axis=2)
        # 根据最近距离分配标签
        new_labels = np.argmin(distances, axis=0)
        # 检查标签是否变化，如果没有变化，则退出循环
        if np.array_equal(new_labels, labels):
            break
        # 更新质心
        new_centroids = np.array([data[new_labels == j].mean(axis=0) for j in range(n_clusters)])
        # 检查质心是否变化
        centroids = new_centroids
        labels = new_labels
    
    return centroids, labels

def predict(data, centroids):
    # 计算 data 到质心的距离
    distances = np.sqrt((data - centroids[:, np.newaxis])**2).sum(axis=2)
    # 分配每个点到最近的质心
    labels = np.argmin(distances, axis=0)
    return labels

# 计算 SSE 函数
def compute_sse(data, centroids, labels):
    sse = 0.0
    for i in range(len(data)):
        # 找到离当前点最近的质心
        closest_centroid_index = labels[i]
        # 累加距离平方和
        sse += np.sum((data[i] - centroids[closest_centroid_index])**2)
    return sse

def main():
  # 数据预处理
  anime_data = pd.read_csv(r'anime.csv')
  features_columns = ['Popularity', 'Score-2', 'Score-3', 'Score-4', 'Score-5', 'Score-6', 'Score-7', 'Score-8', 'Score-9', 'Score-10']

  # 将 'Unknown' 转换为 NaN，并删除包含 NaN 的行
  for feature in features_columns:
      anime_data[feature] = pd.to_numeric(anime_data[feature], errors='coerce')
  anime_data = anime_data.dropna(subset=features_columns)


  # 数据划分
  num_per_class = 60

  # 假设Popularity列的值已经离散化为高、中、低三类
  anime_data = anime_data.sort_values(by='Popularity', ascending=False)
  mid = anime_data.shape[0] // 2
  high = anime_data.iloc[num_per_class: 2 * num_per_class].copy()  # 高流行度
  middle = anime_data.iloc[mid: mid + num_per_class].copy()        # 中流行度
  low = anime_data.tail(num_per_class).copy()                      # 低流行度

  # 为每个类别添加标签
  high['label'] = 0
  middle['label'] = 1
  low['label'] = 2

  # 合并数据
  anime_data_selected = pd.concat([high, middle, low])

  # 归一化处理
  scaler = MinMaxScaler()
  anime_data_selected[features_columns] = scaler.fit_transform(anime_data_selected[features_columns])
  features_normalized = scaler.transform(anime_data[features_columns])

  # Kmeans
  centroids, labels = kmeans(features_normalized, n_clusters=3, max_iter=2000)
  # 对 anime_data_selected[features_columns] 进行预测
  anime_data_selected['Cluster'] = predict(anime_data_selected[features_columns].to_numpy(), centroids)

  # 对于相同的Cluster，计算Popularity的平均值，并重新从高到低分配Cluster的标签
  sorted_popularity_means = anime_data_selected.groupby('Cluster')['Popularity'].mean().sort_values(ascending=False)

  # 创建一个映射，将每个簇标签映射为新标签
  reassign_mapping = {old_label: new_label for old_label, new_label in zip(sorted_popularity_means.index, range(3))}

  # 使用映射重新分配Cluster标签
  anime_data_selected['Cluster'] = anime_data_selected['Cluster'].map(reassign_mapping)


  # 两种方法准确度评价
  accuracy = (anime_data_selected['label'] == anime_data_selected['Cluster']).mean()
  sse_distances = compute_sse(anime_data_selected[features_columns].to_numpy(), centroids, anime_data_selected['Cluster'].to_numpy())

  print(f"Accuracy: {accuracy}\nSSE Distances: {sse_distances}")

  # 选择Score10和Score2两个维度进行可视化
  plt.figure(figsize=(10, 6))
  colors = ['r', 'g', 'b']
  for i in range(3):
      plt.scatter(anime_data_selected[anime_data_selected['Cluster'] == i]['Score-2'],
                  anime_data_selected[anime_data_selected['Cluster'] == i]['Score-10'],
                  c=colors[i], label=f'Cluster {i}')

  plt.title(f'K-means Clustering of Anime Dataset, Accuracy: {accuracy:.4f}, SSE Distances: {sse_distances:.4f}')
  plt.xlabel('Score 2')
  plt.ylabel('Score 10')
  plt.legend()
  plt.show()

if __name__ == '__main__':
    main()