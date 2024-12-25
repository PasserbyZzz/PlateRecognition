import cv2
import numpy as np

SZ = 20  # 字符图片长宽
MIN_AREA = 200  # 车牌区域允许最小面积

def find_waves(threshold, histogram):
    """根据设定的阈值和图片直方图，找出波峰，用于分隔字符"""
    up_point = -1  # 上升点，记录当前上升沿的位置
    is_peak = False  # 标记是否处于波峰状态
    wave_peaks = []  # 存储找到的波峰，每个波峰是一个 (start, end) 的元组
    
    # 如果直方图的第一个元素大于阈值，则认为是从一个波峰开始
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    
    # 遍历直方图的所有元素
    for i, x in enumerate(histogram):
        # 当前处于波峰状态且当前值小于阈值时，认为波峰结束
        if is_peak and x < threshold:
            # 检查波峰宽度是否足够大（至少有2个像素宽），以避免误判小噪声为波峰
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        
        # 当前不处于波峰状态且当前值大于等于阈值时，认为新的波峰开始
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    
    # 处理最后一个波峰，如果存在的话
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))

    #将间隔较小的两个峰合起来，防止左右结构的汉字被分割
    for i in range(len(wave_peaks)-2):
        if wave_peaks[i+1][0] - wave_peaks[i][1] <15:
            new = (wave_peaks[i][0],wave_peaks[i+1][1])
            del wave_peaks[i]
            del wave_peaks[i]
            wave_peaks.insert(i,new)

    #较小的峰去掉
    for i in wave_peaks:
        if i[1]-i[0]< 15:
            wave_peaks.remove(i)
    
    return wave_peaks
    

def separate_characters(card_img, color="blue"):
    """从车牌图像中分离字符
    提供的card_img必须为切割好的方正的车牌照片"""
    
    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
    if color in ["green", "yellow"]:
        gray_img = cv2.bitwise_not(gray_img)
    _, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

    # 查找垂直直方图波峰
    row_num, col_num = gray_img.shape[:2]
    # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
    gray_img = gray_img[3:row_num - 12]     #3,12可以根据效果改
    y_histogram = np.sum(gray_img, axis=0)
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 5  # 对于U和0等字符，阈值应偏小
    
    wave_peaks = find_waves(y_threshold, y_histogram)
    print(wave_peaks)

    if len(wave_peaks) <= 6:
        print("波峰数量过少，可能不是车牌")
        return []

    # 判断是否是左侧车牌边缘
    if wave_peaks and wave_peaks[0][1] - wave_peaks[0][0] < (wave_peaks[-1][1] - wave_peaks[-1][0]) / 3 and wave_peaks[0][0] == 0:
        wave_peaks.pop(0)

    # 组合分离汉字（假设第一个字符是汉字）
    if len(wave_peaks) > 2:
        cur_dis = 0
        for i, wave in enumerate(wave_peaks):
            if wave[1] - wave[0] + cur_dis > (wave_peaks[-1][1] - wave_peaks[-1][0]) * 0.6:
                break
            else:
                cur_dis += wave[1] - wave[0]
        if i > 0:
            wave = (wave_peaks[0][0], wave_peaks[i][1])
            wave_peaks = wave_peaks[i + 1:]
            wave_peaks.insert(0, wave)

    # 去除车牌上的分隔点
    if len(wave_peaks) > 2:
        point = wave_peaks[2]
        if point[1] - point[0] < (wave_peaks[-1][1] - wave_peaks[-1][0]) / 3:
            point_img = gray_img[:, point[0]:point[1]]
            if np.mean(point_img) < 255 / 5:
                wave_peaks.pop(2)

    if len(wave_peaks) <= 6:
        print("波峰数量过少，可能不是车牌")
        return []

    part_cards = [gray_img[:, wave[0]:wave[1]] for wave in wave_peaks]

    # 可视化分割结果（可选）
    for idx, part_card in enumerate(part_cards):
        cv2.imwrite(f'./character_{idx}.jpg', part_card)

    return part_cards


# 示例用法
if __name__ == "__main__":
    image_path = "2.jpg" 
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
    else:
        characters = separate_characters(img, color="blue")  # 指定车牌颜色
        if characters:
            print(f"成功分割出 {len(characters)} 个字符")