import numpy as np
import cv2
import os
import time
from collections import defaultdict

# Step 1: 图像的表示
def extract_color_histogram(image_path):
    """提取图像的颜色直方图并归一化"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def quantize_features(hist):
    """将颜色直方图量化为离散值"""
    quantized = []
    for value in hist:
        if value < 0.3:
            quantized.append(0)
        elif value < 0.6:
            quantized.append(1)
        else:
            quantized.append(2)
    return quantized

# Step 2: 哈希函数计算
def hash_function(quantized_vector, C, hash_planes):
    """计算哈希值"""
    hash_code = []
    for i, plane in enumerate(hash_planes, start=1):
        count_ones = sum(1 for idx in plane if quantized_vector[idx - 1] <= C * (i - 1))
        hash_code.append(1 if count_ones > 0 else 0) 
    return tuple(hash_code)

# Step 3: LSH检索
def insert_to_buckets(quantized_vectors, hash_planes, C):
    """将量化向量插入哈希桶"""
    buckets = defaultdict(list)
    for idx, q_vector in enumerate(quantized_vectors):
        hash_code = hash_function(q_vector, C, hash_planes)
        buckets[hash_code].append(idx + 1)
    return buckets

def query_lsh(target_vector, hash_planes, buckets, C):
    """通过LSH检索相似图片"""
    target_hash_code = hash_function(target_vector, C, hash_planes)
    return buckets.get(target_hash_code, [])

# 最近邻算法（暴力搜索）
def query_nn(target_vector, database_vectors):
    """使用最近邻算法暴力搜索"""
    distances = [np.linalg.norm(target_vector - db_vector) for db_vector in database_vectors]
    closest_idx = np.argmin(distances)
    return closest_idx, distances[closest_idx]

# 加载图像并计算颜色直方图
data_folder = "dataset"  
num_images = 40  
histograms = []
quantized_vectors = []

# 提取所有图像的特征
for i in range(1, num_images + 1):
    image_path = os.path.join(data_folder, f"{i}.jpg")
    histogram = extract_color_histogram(image_path)
    quantized = quantize_features(histogram)
    histograms.append(histogram)
    quantized_vectors.append(quantized)

# 提取目标图像的特征
target_image_path = "target.jpg"
target_histogram = extract_color_histogram(target_image_path)
target_quantized = quantize_features(target_histogram)

# 定义LSH参数
hash_planes = [
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [10, 11, 12]
]

C = 2

buckets = insert_to_buckets(quantized_vectors, hash_planes, C)

# Step 4: 对比LSH和NN搜索
# (1) LSH搜索
start_time = time.time()
similar_image_ids = query_lsh(target_quantized, hash_planes, buckets, C)
lsh_time = time.time() - start_time

if similar_image_ids:
    # 计算目标图像和当前图像之间的距离
    distances = {}
    for img_id in similar_image_ids:
        candidate_hist = extract_color_histogram(os.path.join(data_folder, f"{img_id}.jpg"))
        distances[img_id] = np.linalg.norm(candidate_hist - target_histogram)
    
    # 找到距离最近的匹配
    lsh_result = min(distances, key=distances.get)

# (2) NN搜索
start_time = time.time()
nn_result_idx, nn_distance = query_nn(target_histogram, histograms)
nn_time = time.time() - start_time

# 输出对比结果
print("LSH 搜索：")
print(f"检索时间: {lsh_time:.6f} 秒")
print(f"第{lsh_result}为最相似的图片！")
print("-" * 50)
print("NN 搜索：")
print(f"检索时间: {nn_time:.6f} 秒")
print(f"第{nn_result_idx + 1}为最相似的图片！")
print("-" * 50)