import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据集
train_set = pd.read_csv(r'train_set.csv')
test_set = pd.read_csv(r'test_set.csv')
anime = pd.read_csv(r'anime.csv')

# 构建用户-动漫评分效用矩阵
utility_matrix = train_set.pivot(index='user_id', columns='anime_id', values='rating')
utility_matrix = utility_matrix.fillna(0)# 计算动漫类型的TF-IDF矩阵

# 计算动漫的 TF-IDF 特征矩阵
# 初始化TF-IDF向量器，并去除英文停用词
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(anime['Genres'].fillna(''))

# 构建动漫与索引的映射关系
anime_id_to_index = {anime_id: idx for idx, anime_id in enumerate(anime['Anime_id'])}

# 计算余弦相似度矩阵，输出相似度矩阵的维度
cosine_sim = cosine_similarity(tfidf_matrix)# 计算动漫之间的余弦相似度
print(cosine_sim.shape[0])

# 定义评分预测函数
def predict_ratings(user_id, utility_matrix, cosine_sim, n_recommendations):
    user_ratings = utility_matrix.loc[user_id]  # 获取用户评分情况
    unrated_anime_ids = user_ratings[user_ratings == 0].index.tolist()  # 用户未评分的动漫

    predicted_ratings = {} # 初始化预测评分字典
    for anime_id in unrated_anime_ids:
        #相似度总和， 加权评分总和
        sim_sum = 0
        weighted_rating_sum = 0
        anime_idx = anime_id_to_index.get(anime_id)  # 获取动漫在矩阵中的列索引
        for rated_anime_id, rating in user_ratings.items():
            if rating > 0:  # 只考虑用户已评分的动漫
                rated_anime_idx = anime_id_to_index.get(rated_anime_id)  # 获取已评分动漫在矩阵中的列索引
                sim = cosine_sim[anime_idx, rated_anime_idx]  # 使用索引在相似度矩阵中获取相似度值
                sim_sum += sim
                weighted_rating_sum += sim * rating
        if sim_sum > 0:
            predicted_ratings[anime_id] = weighted_rating_sum / sim_sum
        else:
            predicted_ratings[anime_id] = 0

    # 选取预测评分最高的前 n 个动漫进行推荐
    top_n_anime = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return top_n_anime

# 定义计算 SSE 的函数
def calculate_sse(test_set, utility_matrix, cosine_sim):
    sse = 0
    for idx, row in test_set.iterrows():
        user_id = row['user_id']
        anime_id = row['anime_id']
        true_rating = row['rating']
        predicted_rating = predict_ratings(user_id, utility_matrix, cosine_sim, 1)[0][1]
        sse += (true_rating - predicted_rating) ** 2
    return sse

# 指定用户和推荐数量
user_id = 629
n_recommendations = 20

# 执行推荐和评估
top_n_anime_recommendations = predict_ratings(user_id, utility_matrix, cosine_sim, n_recommendations)
sse = calculate_sse(test_set, utility_matrix, cosine_sim)

with open('ItemCF_result.txt', 'w', encoding='utf-8') as f:
    for anime_id, rating in top_n_anime_recommendations:
        f.write(f"Anime ID: {anime_id} Predicted Rating: {rating}\n")
    f.write(f'SSE: {sse}')
