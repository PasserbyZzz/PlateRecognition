import numpy as np
import cv2
import os
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

# Step 4: 根据哈希桶查询图片
def query_image(target_vector, hash_planes, buckets, C):
    """通过LSH检索相似图片"""
    target_hash_code = hash_function(target_vector, C, hash_planes)
    return buckets.get(target_hash_code, [])

# 加载图像，并计算颜色直方图
data_folder = "dataset"
num_images = 40
quantized_vectors = []

for i in range(1, num_images + 1):
    image_path = os.path.join(data_folder, f"{i}.jpg")
    histogram = extract_color_histogram(image_path)
    quantized = quantize_features(histogram)
    quantized_vectors.append(quantized)

# 计算颜色直方图并量化
target_image_path = "target.jpg"
target_histogram = extract_color_histogram(target_image_path)
target_quantized = quantize_features(target_histogram)

# 随机定义哈希投影集合
hash_planes = [
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [10, 11, 12]
]

C = 2

# 插入数据到哈希桶
buckets = insert_to_buckets(quantized_vectors, hash_planes, C)

# 查询目标图片的相似图片
similar_image_ids = query_image(target_quantized, hash_planes, buckets, C)

# 根据欧几里得距离验证并精确匹配
if similar_image_ids:
    # 计算目标图像和当前图像之间的距离
    distances = {}
    for img_id in similar_image_ids:
        candidate_hist = extract_color_histogram(os.path.join(data_folder, f"{img_id}.jpg"))
        distances[img_id] = np.linalg.norm(candidate_hist - target_histogram)
    
    # 找到距离最近的匹配
    closest_match = min(distances, key=distances.get)
    print(f"最相似的图片是第{closest_match}张")
    print(f"与目标图像间的距离为：{distances[closest_match]}")
else:
    print("没有找到相似的图像！")
