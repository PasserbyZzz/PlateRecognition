import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# 正确显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取图像
img1 = cv2.imread('images/img1.jpg')
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('images/img2.jpg')
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img3 = cv2.imread('images/img3.jpg')
img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# 计算每个通道的总能量并归一化
def calculate_color_energy(image_rgb):
    color_totals = [np.sum(image_rgb[..., i]) for i in range(3)]
    total_sum = sum(color_totals)
    normalized_totals = [x / total_sum for x in color_totals]
    return normalized_totals

# 绘制颜色能量柱状图
def plot_color_energy(ax, color_energy, title):
    colors = ['red', 'green', 'blue']
    ax.bar(range(3), color_energy, color=colors, tick_label=['R', 'G', 'B'])
    for i, v in enumerate(color_energy):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    ax.set_ylim(0, 1)
    ax.set_title(title)

# 计算梯度图
def compute_gradient(image):
    # 将图像转换为浮点型以避免溢出
    image = image.astype(np.float32)
    
   # X方向梯度
    grad_x = np.zeros_like(image)
    grad_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
    
    # Y方向梯度
    grad_y = np.zeros_like(image)
    grad_y[1:-1, :] = image[2:, :] - image[:-2, :]
    
    # 梯度幅值 M(x, y)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude[1:-1, 1:-1]

img1_grad = compute_gradient(img1_gray)
img2_grad = compute_gradient(img2_gray)
img3_grad = compute_gradient(img3_gray)

# 计算并绘制直方图
def plot_histograms(image_rgb, image_gray, image_grad, ax_rgb_hist, ax_gray_hist, ax_grad_hist):

    color_energy1 = calculate_color_energy(image_rgb)
    plot_color_energy(ax_rgb_hist, color_energy1, "颜色直方图")

    # 灰度直方图
    ax_gray_hist.hist(image_gray.ravel(), bins=256, color='black', alpha=0.5, range=(0, 256), density=True)
    ax_gray_hist.set_title("灰度直方图")

    # 梯度直方图
    ax_grad_hist.hist(image_grad.ravel(), bins=361, color='gray', alpha=0.5, range=(0, 361), density=True)
    ax_grad_hist.set_title("梯度直方图")

# 绘制颜色直方图、灰度直方图和梯度直方图
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
plot_histograms(img1_rgb, img1_gray, img1_grad, axes[0, 0], axes[0, 1], axes[0, 2])
plot_histograms(img2_rgb, img2_gray, img2_grad, axes[1, 0], axes[1, 1], axes[1, 2])
plot_histograms(img3_rgb, img3_gray, img3_grad, axes[2, 0], axes[2, 1], axes[2, 2])

# 保存图像
plt.tight_layout()
plt.savefig('image_histograms.png')
plt.close()