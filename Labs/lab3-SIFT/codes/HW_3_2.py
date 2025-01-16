import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 加载图像
target_path = 'target.jpg'  # 目标图像路径
target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

# 实例化SIFT对象
sift = cv2.SIFT_create()

# 提取目标图像的关键点和描述子
kp_target, des_target = sift.detectAndCompute(target_img, None)

# 创建BFMatcher对象
# 使用crossCheck=False以允许多对匹配
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  

# 遍历数据集图像，进行匹配并保存每一幅图
image_paths = ['dataset/1.jpg', 'dataset/2.jpg', 'dataset/3.jpg', 'dataset/4.jpg', 'dataset/5.jpg']
for i, image_path in enumerate(image_paths):
    # 加载数据集图像
    dataset_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 提取数据集图像的关键点和描述子
    kp_dataset, des_dataset = sift.detectAndCompute(dataset_img, None)

    # 使用KNN进行特征匹配（K=2）
    knn_matches = bf.knnMatch(des_target, des_dataset, k=2)

    # 通过比值测试筛选好的匹配
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:  # 比值测试，保留好的匹配
            good_matches.append(m)

    # 绘制好的匹配结果
    img_matches = cv2.drawMatches(target_img, kp_target, dataset_img, kp_dataset, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存每一幅匹配图像
    match_filename = f"output/HW_3_2/match_with_{os.path.basename(image_path)}.png"
    plt.imshow(img_matches)
    plt.axis('off')  
    plt.savefig(match_filename, bbox_inches='tight', pad_inches=0) 