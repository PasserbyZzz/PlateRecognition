import cv2
import numpy as np

# 检测图像中的蓝色或者绿色区域来检测是否有车牌
def detect_blue_regions(image_path):
    # Step 1: 读取图像并转换为 HSV 色彩空间
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: 设置蓝色的 HSV 范围
    lower_blue = np.array([100, 150, 50])  # 蓝色的下界
    upper_blue = np.array([140, 255, 255])  # 蓝色的上界

    # Step 3: 创建掩码并提取蓝色区域
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_regions = cv2.bitwise_and(image, image, mask=mask)

    # Step 4: 寻找蓝色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: 绘制蓝色区域的轮廓
    result_image = image.copy()
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 1000:  # 过滤小区域
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制绿色矩形框

    # 显示结果
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Blue Regions", blue_regions)
    cv2.imshow("Detected Blue Areas", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 测试
image_path = "dataset/Blue/5.jpg"
detect_blue_regions(image_path)