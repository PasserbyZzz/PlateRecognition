import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置 Matplotlib 字体
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

# Roberts 算子
def compute_gradient_with_roberts(image):
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    Gx = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_x)
    Gy = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_y)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

# Prewitt 算子
def compute_gradient_with_prewitt(image):
    prewitt_kernel_x = np.array([[-1, 0, 1], 
                                 [-1, 0, 1], 
                                 [-1, 0, 1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[1, 1, 1], 
                                 [0, 0, 0], 
                                 [-1, -1, -1]], dtype=np.float32)
    Gx = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_x)
    Gy = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_y)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

# 非极大值抑制
def non_maximum_suppression(magnitude, direction):
    magnitude = (magnitude / magnitude.max()) * 255
    nms = np.zeros_like(magnitude, dtype=np.float32)
    angle = direction * 180 / np.pi
    angle[angle < 0] += 180
    rows, cols = magnitude.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

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

# 主函数
def canny_edge_detection_with_gradient(image_path, method, low_threshold, high_threshold):
    gray = read_img(image_path)
    blurred = apply_gaussian_blur(gray)
    
    if method == "roberts":
        magnitude, direction = compute_gradient_with_roberts(blurred)
    elif method == "prewitt":
        magnitude, direction = compute_gradient_with_prewitt(blurred)
    else:
        magnitude, direction = compute_gradient_with_prewitt(blurred)
    
    nms = non_maximum_suppression(magnitude, direction)
    result, strong_edge, weak_edge = double_threshold(nms, low_threshold, high_threshold)
    final_result = hysteresis_tracking(result, strong_edge, weak_edge)
    return final_result

# OpenCV 内置 Canny 算子
def canny_with_opencv(image_path, low_threshold, high_threshold):
    gray = read_img(image_path)
    return cv2.Canny(gray, low_threshold, high_threshold)

# 参数设置
image_path = "dataset/1.jpg"
low_threshold, high_threshold = 10, 25

# 计算三种边缘检测结果
roberts_result = canny_edge_detection_with_gradient(image_path, "roberts", low_threshold, high_threshold)
prewitt_result = canny_edge_detection_with_gradient(image_path, "prewitt", low_threshold, high_threshold)
opencv_result = canny_with_opencv(image_path, low_threshold, high_threshold)

# 显示结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(roberts_result, cmap='gray')
plt.title("Roberts 算子边缘检测")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(prewitt_result, cmap='gray')
plt.title("Prewitt 算子边缘检测")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(opencv_result, cmap='gray')
plt.title("OpenCV Canny 算子边缘检测")
plt.axis('off')

plt.tight_layout()
plt.savefig('output/HW_2_3')
