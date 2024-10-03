import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 读取数据
train_set = pd.read_csv(r'train_set.csv')
test_set = pd.read_csv(r'test_set.csv')

# 构建用户-动漫效用矩阵
utility_matrix = train_set.pivot(index='user_id', columns='anime_id', values='rating')
utility_matrix = utility_matrix.fillna(0) # 将缺失值填充为0

# 计算用户之间的皮尔逊相关系数，生成用户相似度矩阵
similarity_matrix = utility_matrix.T.corr(method='pearson')

# 预测评分
def predict_rating(user_id, anime_id, k):
    if anime_id not in utility_matrix.columns:
        return 0  # 如果动漫不在训练集中，返回0

    # 找到与当前用户最相似的k个用户
    similar_users = similarity_matrix.loc[user_id].sort_values(ascending=False).index
    # 保证选出k个不是自己且对anime_id已评分的用户
    similar_users = similar_users[(similar_users != user_id) & (utility_matrix.loc[similar_users, anime_id] != 0)]
    similar_users = similar_users[:k]
    # print(similar_users)
    if(len(similar_users)<k) :
        return 0  # 如果相似用户数量少于k，返回0
    
    numerator = 0  # 分子初始化
    denominator = 0   # 分母初始化
    for similar_user_id in similar_users:
        # 计算分子：相似度和评分的乘积的累加
        numerator += similarity_matrix.loc[user_id, similar_user_id] * utility_matrix.loc[similar_user_id, anime_id]
        # 计算分母:相似度累加
        denominator += similarity_matrix.loc[user_id, similar_user_id] 
    if denominator == 0:
        return 0
    
    res = numerator / denominator
    return res

# 评估算法
def evaluate(test_set, k):
    sse = 0

    for idx, row in test_set.iterrows():
        user_id = row['user_id']
        anime_id = row['anime_id']
        true_rating = row['rating']
        pred_rating = predict_rating(user_id, anime_id, k)
        sse += (true_rating - pred_rating) ** 2

    return sse

# 找到用户629的未评分动漫
user_id = 629
user_ratings = utility_matrix.loc[user_id]
unrated_anime = user_ratings[user_ratings == 0].index.tolist()

# 对这些未评分动漫进行评分预测
k = 3
predicted_ratings = {}

for anime_id in unrated_anime:
    predicted_ratings[anime_id] = predict_rating(user_id, anime_id, k)

# 选取评分最高的20个动漫进行推荐
top_20_anime = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:20]

sse = evaluate(test_set, k)

with open('UserCF_result.txt', 'w', encoding="utf-8") as f:
    f.write('对用户推荐如下电影：\n')
    f.write('Anime  Predicted Rating\n')
    for anime_id, rating in top_20_anime:
        f.write(f'{anime_id},  {rating:.2f}\n')
    f.write(f'SSE: {sse}')
