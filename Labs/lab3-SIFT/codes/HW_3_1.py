import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 构建图像金字塔
def build_pyramid(image, levels=3):
    pyramid = [image]
    for i in range(1, levels):
        scaled_image = cv2.pyrDown(pyramid[-1])  # 图像降采样
        pyramid.append(scaled_image)
    return pyramid

# Harris角点检测
def detect_keypoints(image, max_corners=5000, quality_level=0.01, min_distance=5):
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    if corners is None:
        return []
    # 转换角点为 OpenCV 的 KeyPoint 对象
    keypoints = [cv2.KeyPoint(pt[0][0], pt[0][1], 1) for pt in corners]
    return keypoints

# 计算SIFT描述子
def compute_sift_descriptor(image, keypoints):
    descriptors = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch_size = 16
        half_size = patch_size // 2

        # 提取16x16邻域
        if y - half_size < 0 or x - half_size < 0 or y + half_size > image.shape[0] or x + half_size > image.shape[1]:
            continue  # 跳过超出边界的关键点

        patch = image[y - half_size:y + half_size, x - half_size:x + half_size]
        descriptor = np.zeros((4, 4, 8))

        sub_size = patch_size // 4
        for i in range(4):
            for j in range(4):
                sub_patch = patch[i * sub_size:(i + 1) * sub_size, j * sub_size:(j + 1) * sub_size]

                dx = cv2.Sobel(sub_patch, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(sub_patch, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx) * 180 / np.pi
                angle[angle < 0] += 360

                hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
                descriptor[i, j, :] = hist

        descriptor = descriptor.flatten()
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        descriptors.append(descriptor)
    return np.array(descriptors, dtype=np.float32)  # 确保类型为float32

# 匹配target.jpg并绘制结果

target_path = "target.jpg"
dataset_dir = "dataset"
output_dir = "output"  # 保存匹配结果的文件夹

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取目标图像
target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

# 使用自己编写的函数检测关键点和计算描述子
target_kp = detect_keypoints(target_img)
target_des = compute_sift_descriptor(target_img, target_kp)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # 使用 L2 范数进行匹配
for filename in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 检测关键点和描述子
    kp = detect_keypoints(img)
    des = compute_sift_descriptor(img, kp)

    # KNN特征匹配
    matches = bf.knnMatch(target_des, des, k=2)

    # 筛选匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    result_img = cv2.drawMatches(target_img, target_kp, img, kp, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配结果
    plt.figure(figsize=(12, 6))
    plt.imshow(result_img[:, :, ::-1])  
    plt.axis("off")
    output_path = os.path.join(output_dir, f"HW_3_1/match_{filename}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()