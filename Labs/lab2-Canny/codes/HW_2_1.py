import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取图像并转为灰度图
def read_img(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# 高斯模糊
def apply_gaussian_blur(gray, kernel_size=5, sigma=1.4):
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    return blurred

# 非极大值抑制
def non_maximum_suppression(magnitude, direction):
    magnitude = (magnitude / magnitude.max()) * 255  # 归一化到0-255
    nms = np.zeros_like(magnitude, dtype=np.float32)
    angle = direction * 180 / np.pi  # 转为角度
    angle[angle < 0] += 180  # 修正负角度为正角度

    rows, cols = magnitude.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # 根据角度划分邻域
            q = r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, min(j + 1, cols - 1)] * (abs(angle[i, j] / 22.5)) + \
                    magnitude[i, min(j + 2, cols - 1)] * (1 - abs(angle[i, j] / 22.5))
                r = magnitude[i, max(j - 1, 0)] * (abs(angle[i, j] / 22.5)) + \
                    magnitude[i, max(j - 2, 0)] * (1 - abs(angle[i, j] / 22.5))
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[min(i + 1, rows - 1), max(j - 1, 0)] * (abs((angle[i, j] - 45) / 22.5)) + \
                    magnitude[min(i + 2, rows - 1), max(j - 2, 0)] * (1 - abs((angle[i, j] - 45) / 22.5))
                r = magnitude[max(i - 1, 0), min(j + 1, cols - 1)] * (abs((angle[i, j] - 45) / 22.5)) + \
                    magnitude[max(i - 2, 0), min(j + 2, cols - 1)] * (1 - abs((angle[i, j] - 45) / 22.5))
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[min(i + 1, rows - 1), j] * (abs((angle[i, j] - 90) / 22.5)) + \
                    magnitude[min(i + 2, rows - 1), j] * (1 - abs((angle[i, j] - 90) / 22.5))
                r = magnitude[max(i - 1, 0), j] * (abs((angle[i, j] - 90) / 22.5)) + \
                    magnitude[max(i - 2, 0), j] * (1 - abs((angle[i, j] - 90) / 22.5))
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[max(i - 1, 0), max(j - 1, 0)] * (abs((angle[i, j] - 135) / 22.5)) + \
                    magnitude[max(i - 2, 0), max(j - 2, 0)] * (1 - abs((angle[i, j] - 135) / 22.5))
                r = magnitude[min(i + 1, rows - 1), min(j + 1, cols - 1)] * (abs((angle[i, j] - 135) / 22.5)) + \
                    magnitude[min(i + 2, rows - 1), min(j + 2, cols - 1)] * (1 - abs((angle[i, j] - 135) / 22.5))

            # 非极大值抑制
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0

    return nms

# 双阈值检测
def double_threshold(nms, low_threshold, high_threshold):
    strong_edge = 255
    weak_edge = 50
    result = np.zeros_like(nms)

    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))

    result[strong_i, strong_j] = strong_edge
    result[weak_i, weak_j] = weak_edge

    return result, strong_edge, weak_edge

# 边缘连接
def hysteresis_tracking(result, strong_edge, weak_edge):
    rows, cols = result.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i, j] == weak_edge:
                if ((result[i+1, j-1:j+2] == strong_edge).any() or
                    (result[i-1, j-1:j+2] == strong_edge).any() or
                    (result[i, [j-1, j+1]] == strong_edge).any()):
                    result[i, j] = strong_edge
                else:
                    result[i, j] = 0
    return result

# 主函数：Canny 边缘检测
def canny_edge_detection(image_path, low_threshold, high_threshold):
    # 读取图像并转为灰度图
    gray = read_img(image_path)

    # 高斯模糊
    blurred = apply_gaussian_blur(gray)

    # Sobel 算子计算梯度强度和方向
    Gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)

    # 非极大值抑制
    nms = non_maximum_suppression(magnitude, direction)

    # 双阈值检测
    result, strong_edge, weak_edge = double_threshold(nms, low_threshold, high_threshold)

    # 边缘连接
    final_result = hysteresis_tracking(result, strong_edge, weak_edge)

    return final_result

# 图片路径和阈值
image_paths = ["dataset/1.jpg", "dataset/2.jpg", "dataset/3.jpg"]
thresholds = [[10, 25], [10, 25], [10, 25]]  # 每张图片的低阈值和高阈值
thresholds_cv = [[50, 125], [60, 150], [50, 125]]

# 处理图片并显示对比
plt.figure(figsize=(12, 8))

for i in range(3):
    image_path = image_paths[i]
    low_threshold, high_threshold = thresholds[i]
    low_threshold_cv, high_threshold_cv = thresholds_cv[i]

    # 自定义 Canny 实现
    custom_result = canny_edge_detection(image_path, low_threshold, high_threshold)

    # OpenCV Canny 实现
    gray = read_img(image_path)
    opencv_result = cv2.Canny(gray, low_threshold_cv, high_threshold_cv)

    # 显示结果
    plt.subplot(2, 3, i+1)
    plt.imshow(custom_result, cmap='gray')
    plt.title(f"纯手搓的Canny检测")
    plt.axis('off')

    plt.subplot(2, 3, i+4)
    plt.imshow(opencv_result, cmap='gray')
    plt.title(f"OpenCV自带的Canny检测")
    plt.axis('off')

plt.tight_layout()
plt.show()
# plt.savefig('output/HW_2_1')