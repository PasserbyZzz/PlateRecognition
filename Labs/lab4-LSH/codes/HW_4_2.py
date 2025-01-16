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
        buckets[hash_code].append(idx + 1)  # 图片编号从1开始
    return buckets

def query_image(target_vector, hash_planes, buckets, C):
    """检索目标图像的相似图片"""
    target_hash_code = hash_function(target_vector, C, hash_planes)
    return buckets.get(target_hash_code, [])

# 加载图像并计算颜色直方图
data_folder = "dataset"  
num_images = 40  
quantized_vectors = []

# 提取所有图像的特征
for i in range(1, num_images + 1):
    image_path = os.path.join(data_folder, f"{i}.jpg")
    histogram = extract_color_histogram(image_path)
    quantized = quantize_features(histogram)
    quantized_vectors.append(quantized)

# 提取目标图像的特征
target_image_path = "target.jpg"  
target_histogram = extract_color_histogram(target_image_path)
target_quantized = quantize_features(target_histogram)

# 定义多组投影集合
hash_planes_set = [
    [[1, 3, 5], [2, 4, 6], [7, 8, 9], [10, 11, 12]],  # 投影集合1
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],  # 投影集合2
    [[1, 6, 7], [2, 8, 9], [3, 4, 5], [10, 11, 12]]   # 投影集合3
]

C = 2  # 哈希参数
correct_image_id = 38  # 目标图片的真实匹配为第38张

results = []

# 对不同投影集合分别进行检索
for i, hash_planes in enumerate(hash_planes_set):
    start_time = time.time()

    # 插入数据到哈希桶
    buckets = insert_to_buckets(quantized_vectors, hash_planes, C)

    # 查询目标图片
    similar_image_ids = query_image(target_quantized, hash_planes, buckets, C)

    # 检索时间
    elapsed_time = time.time() - start_time

    # 计算检索准确性
    if similar_image_ids:
        distances = {
            img_id: np.linalg.norm(
                extract_color_histogram(os.path.join(data_folder, f"{img_id}.jpg")) - target_histogram
            )
            for img_id in similar_image_ids
        }
        closest_match = min(distances, key=distances.get)
        accuracy = 1.0 if closest_match == correct_image_id else 0.0
    else:
        accuracy = 0.0

    # 记录结果
    results.append({
        "投影集合": f"投影集合{i + 1}",
        "检索时间(s)": elapsed_time,
        "检索准确率": accuracy,
        "检索结果": closest_match
    })

# 输出结果
for result in results:
    print(f"投影集合: {result['投影集合']}")
    print(f"检索时间: {result['检索时间(s)']:.4f}秒")
    print(f"检索准确率: {result['检索准确率'] * 100:.2f}%")
    print(f"检索结果: 第{result['检索结果']}张为最相似的图片！")
    print("-" * 50)
