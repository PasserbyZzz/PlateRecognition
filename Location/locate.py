import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000   # 车牌区域允许最小面积
Max_Area = 9000   # 车牌区域允许最大面积

def point_limit(point):
	'''
	限制点坐标不小于零
	'''
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0
		
def find_waves(threshold, histogram):
    '''
	根据设定的阈值和图片直方图，找出波峰，用于分隔字符
	'''
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

    if wave_peaks[0][0] / wave_peaks[-1][1] < 0.01:
        wave_peaks.remove(wave_peaks[0])
    
    return wave_peaks
	
class PlatesLocator:
	'''
	创建CarLocator类, 用于定位车牌
	'''	
	def __init__(self):
		'''
		初始化函数:车牌识别的部分参数保存在js中,便于根据图片分辨率做调整
		'''
		f = open('./Location/config.js')
		j = json.load(f)
		for c in j["config"]:
			if c["open"]:
				self.cfg = c.copy()
				break
		else:
			raise RuntimeError('没有设置有效配置参数')
		
	def accurate_place(self, card_img_hsv, limit1, limit2, color):
		'''
		精确定位车牌区域；
		输入-> card_img_hsv:HSV图像; limit1, limit2:H的上下限; color:车牌颜色
		返回->xl: 左边界的 x 坐标；右边界的 x 坐标; yh: 上边界的 y 坐标; yl: 下边界的 y 坐标。
		'''
		row_num, col_num = card_img_hsv.shape[:2]
		xl = col_num # 初始化左边界为最大宽度
		xr = 0 # 初始化右边界为 0
		yh = 0 # 初始化上边界为 0
		yl = row_num # 初始化下边界为最大高度

		# 行像素中符合条件的最少个数
		row_num_limit = self.cfg["row_num_limit"]
		# 列像素中符合条件的最少个数
		# 非绿色车牌：列中有至少 80% 的像素符合条件
		# 绿色车牌：有渐变效果
		col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5 

		# 裁剪上下边界
		for i in range(row_num):
			count = 0
			for j in range(col_num):
				# 获取当前像素的HSV值
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)

				# 判断当前像素是否符合目标颜色
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1
			# 达到列阈值
			if count > col_num_limit:
				if yl > i:
					yl = i
				if yh < i and color != 'green': # 绿色车牌不裁上边界
					yh = i

		# 裁剪左右边界
		for j in range(col_num):
			count = 0
			for i in range(row_num):
				# 获取当前像素的HSV值
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)

				# 判断当前像素是否符合目标颜色
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1

			# 达到行阈值
			if count > row_num - row_num_limit:
				if xl > j:
					xl = j
				if xr < j:
					xr = j

		return xl, xr, yh, yl
	
	def locate_plates(self, car_pic, resize_rate=1, para_type="ORIGIN"):
		'''
		主方法，完成车牌检测；
		输入->car_pic:输入图片的路径; resize_rate:图像的缩放比例，用于控制输入图像的大小
		输出->plate_imgs:检测到的车牌区域的图像列表; plate_colors:车牌对应的颜色列表; origin_imgs:原始图像
		'''
		# Step1: 读取和调整图像
		if car_pic == "camera":
			cap = cv2.VideoCapture(0)
			if not cap.isOpened():
				return 0, 0
			
			while True:
				ret, frame = cap.read()
				if not ret:
					return 0, 0
				original_frame = frame.copy()
				
				# 绘制提示词
				text = "Press 'q' to take a picture!"
				font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
				font_scale = 1.2                 # 字体大小
				color = (85, 164, 79)            # 字体颜色 (B, G, R) 格式
				thickness = 5					 # 字体粗细
				# 获取文字的尺寸
				(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
				# 获取图像的宽高
				image_height, image_width = frame.shape[:2]
				# 计算文字的起始坐标
				x = (image_width - text_width) // 2
				y = (image_height + text_height) // 8
				cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

				cv2.imshow("Camera", frame)
				# 按 'q' 手动退出，并保存图片
				if cv2.waitKey(1) & 0xFF == ord('q'): 
					break

			cap.release()
			cv2.destroyAllWindows()
			img = original_frame

		else:
			img = cv2.imread(car_pic)

		original_img = img.copy()

		pic_height, pic_width = img.shape[:2]
		# 限制图片的最大宽度，否则按比例缩小
		if pic_width > MAX_WIDTH:
			pic_rate = MAX_WIDTH / pic_width
			img = cv2.resize(img, (MAX_WIDTH, int(pic_height * pic_rate)), interpolation=cv2.INTER_LANCZOS4)
			pic_height, pic_width = img.shape[:2]

		# 如果指定了缩放比例resize_rate，按比例调整图片的尺寸
		if resize_rate != 1:
			img = cv2.resize(img, (int(pic_width * resize_rate), int(pic_height * resize_rate)), interpolation=cv2.INTER_LANCZOS4)
			pic_height, pic_width = img.shape[:2]
			
		# print(f"height: {pic_height}, width: {pic_width}")

		# Step2: 图像预处理
		blur = self.cfg["blur"]
		# 高斯模糊，降低图像噪声，减少干扰
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0) # 图片分辨率调整

		# 转换为灰度图，为后续边缘检测和二值化做准备
		old_img = img
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# cv2.imwrite('./figures/img_gauss_gray.png', img)

		# 去掉图像中不会是车牌的区域
		if para_type == "ORIGIN":
			kernel = np.ones((20, 20), np.uint8)
		elif para_type == "HIGH":
			kernel = np.ones((15, 15), np.uint8)
		elif para_type == "LOW":
			kernel = np.ones((25, 25), np.uint8)
		# 开运算，去除图片中不相关的小物体
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
		# 加权叠加，将两幅图像合成为一幅图像，增强车牌区域的对比度
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0) 
		
		# cv2.imwrite('./figures/img_opening.png', img_opening)

		# Step3: 边缘检测
		ret_, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 二值化
		img_edge = cv2.Canny(img_thresh, 100, 200) # Canny边缘检测
		# cv2.imwrite('./figures/img_edge.png', img_edge)

		if para_type == "ORIGIN":
			kernel_close = np.ones((4, 20), np.uint8)  # 闭运算核（较宽）
			kernel_open = np.ones((4, 15), np.uint8)   # 开运算核（较小）
		elif para_type == "HIGH":
			kernel_close = np.ones((7, 30), np.uint8)  # 闭运算核（较宽）
			kernel_open = np.ones((4, 12), np.uint8)   # 开运算核（较小）
		elif para_type == "LOW":
			kernel_close = np.ones((4, 10), np.uint8)  # 闭运算核（较宽）
			kernel_open = np.ones((7, 15), np.uint8)   # 开运算核（较小）
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel_close)
		# cv2.imwrite('img_edge_close.png', img_edge1)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel_open)
		# cv2.imwrite('img_edge_close_open.png', img_edge2)

		# # 使用闭运算和开运算让图像边缘成为一个整体
		# kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
		# # 闭运算，连接边缘中的小缺口
		# img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		# # 开运算，消除边缘区域中的小噪声
		# img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
		# img_edge2 = cv2.morphologyEx(img_edge2, cv2.MORPH_OPEN, kernel)
		# # cv2.imwrite('img_edge_close_open.png', img_edge2)

		# Step4: 轮廓检测
		# 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
		try:
			"""
            image：可能是跟输入contour类似的一张二值图；
            contours：list结构，列表中每个元素代表一个边沿信息。每个元素是(x,1,2)的三维向量，x表示该条边沿里共有多少个像素点，
            第三维的那个“2”表示每个点的横、纵坐标；
            hierarchy：返回类型是(x,4)的二维ndarray。x和contours里的x是一样的意思。如果输入选择cv2.RETR_TREE，则以树形
            结构组织输出，hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。
            """
			contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 查找图像中的轮廓
		except ValueError:
			image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# 通过面积过滤掉小的轮廓
		contours = [cnt for cnt in contours if  Min_Area < cv2.contourArea(cnt)] #< Max_Area]

		# print(f"轮廓数量: {len(contours)}")

		# Step5: 矩形筛选
		# 一一排除不是车牌的矩形区域
		car_contours = []
		for cnt in contours:
			rect = cv2.minAreaRect(cnt) # 获取最小外接矩形
			area_width, area_height = rect[1]
			if area_width < area_height:
				area_width, area_height = area_height, area_width
			wh_ratio = area_width / area_height

			#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
			if wh_ratio > 2 and wh_ratio < 5.5:
				car_contours.append(rect)
				# box = cv2.boxPoints(rect)
				# box = np.intp(box)
				# old_img = cv2.drawContours(old_img, [box], 0, (0, 0, 255), 2)
				# cv2.imshow("edge4", old_img)
				# cv2.waitKey(0)
		# print(f"矩形车牌数量: {len(car_contours)}")
		# print("精确定位车牌")

		# Step6: 精确定位车牌
		card_imgs = []
		# 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
		for rect in car_contours:
			# # 创造角度，使得左、高、右、低拿到正确的值
			if rect[2] > -1 and rect[2] < 1:
				angle = 1
			else:
				angle = rect[2]

			# 扩大范围，避免车牌边缘被排除
			rect = (rect[0], (rect[1][0]+4, rect[1][1]+4), angle) 

			# 获取矩形的 4 个顶点
			box = cv2.boxPoints(rect)
			heigth_point = right_point = [0, 0]
			left_point = low_point = [pic_width, pic_height]
			for point in box:
				if left_point[0] > point[0]:
					left_point = point
				if low_point[1] > point[1]:
					low_point = point
				if heigth_point[1] < point[1]:
					heigth_point = point
				if right_point[0] < point[0]:
					right_point = point

			# 正角度
			if left_point[1] <= right_point[1]: 
				new_right_point = [right_point[0], heigth_point[1]]
				if (right_point[0] == heigth_point[0]):
					card_img = old_img[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
					card_imgs.append(card_img)
					continue
				pts2 = np.float32([left_point, heigth_point, new_right_point]) # 字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
				point_limit(new_right_point)
				point_limit(heigth_point)
				point_limit(left_point)
				card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
				card_imgs.append(card_img)
				# cv2.imshow("card", card_img)
				# cv2.waitKey(0)

			# 负角度
			elif left_point[1] > right_point[1]:
				new_left_point = [left_point[0], heigth_point[1]]
				if (left_point[0] == heigth_point[0]):
					card_img = old_img[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
					card_imgs.append(card_img)
					continue
				pts2 = np.float32([new_left_point, heigth_point, right_point]) # 字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
				point_limit(right_point)
				point_limit(heigth_point)
				point_limit(new_left_point)
				card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
				card_imgs.append(card_img)
				# cv2.imshow("card", card_img)
				# cv2.waitKey(0)

		# Step7: 确定车牌颜色
		# 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿车牌
		colors = []
		for card_index, card_img in enumerate(card_imgs):
			green = blue = 0
			try:
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			except BaseException:
				continue
			# 有转换失败的可能，原因来自于上面矫正矩形出错
			if card_img_hsv is None:
				continue
			row_num, col_num= card_img_hsv.shape[:2]
			card_img_count = row_num * col_num

			for i in range(row_num):
				for j in range(col_num):
					H = card_img_hsv.item(i, j, 0)
					S = card_img_hsv.item(i, j, 1)
					V = card_img_hsv.item(i, j, 2)

					# 绿色HSV范围
					if 35 < H <= 99 and S > 43 and V > 46: 
						green += 1
					# 蓝色HSV范围
					elif 99 < H <= 124 and S > 43 and V > 46: 
						blue += 1
			
			# 比较颜色统计结果，根据最多的像素颜色决定区域是否为车牌
			color = "none"
			limit1 = limit2 = 0

			if green * 2 >= card_img_count:
				color = "green"
				limit1 = 35
				limit2 = 99
			elif blue * 2.5 >= card_img_count:
				color = "blue"
				limit1 = 100
				limit2 = 124 # 有的图片有色偏偏紫

			# print(f"车牌颜色: {color}")
			colors.append(color)

			if limit1 == 0:
				continue

			# 根据车牌颜色再定位，缩小边缘非车牌边界
			xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
			if yl == yh and xl == xr:
				continue

			need_accurate = False
			if yl >= yh:
				yl = 0
				yh = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True
			card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]

			# 可能x或y方向未缩小，需要再试一次
			if need_accurate: 
				card_img = card_imgs[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yh, yl = self.accurate_place(card_img_hsv, limit1, limit2, color)
				if yl == yh and xl == xr:
					continue
				if yl >= yh:
					yl = 0
					yh = row_num
				if xl >= xr:
					xl = 0
					xr = col_num
			card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]

		# 筛选出颜色为蓝色或者绿色的车牌
		plate_imgs, plate_colors = [], []
		for color, card_img in zip(colors, card_imgs):
			if color in ("blue", "green"):
				plate_imgs.append(card_img)
				plate_colors.append(color)

		return original_img, plate_imgs, plate_colors
	
	def separate_characters(self, card_img, color="blue"):
		'''
		从车牌图像中分离字符
		输入->card_img:切割好的方正的车牌照片; color:车牌颜色
		输出->part_cards:
		'''
		# 读入灰度图像
		gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
		# 绿车牌字符比背景暗，与蓝车牌刚好相反，所以绿车牌需要反向
		if color == "green":
			gray_img = cv2.bitwise_not(gray_img)
		_, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		# 查找垂直直方图波峰
		row_num, col_num = gray_img.shape[:2]

		gray_img = gray_img[int(row_num*0.02):row_num - int(row_num*0.02)]     # 3,12可以根据效果改
		y_histogram = np.sum(gray_img, axis=0)
		y_min = np.min(y_histogram)
		y_average = np.sum(y_histogram) / y_histogram.shape[0]
		y_threshold = (y_min + y_average) / 20  # 对于U和0等字符，阈值应偏小
		
		wave_peaks = find_waves(y_threshold, y_histogram)
		# print(wave_peaks)

		if len(wave_peaks) <= 5:
			# print("无法进行字符分割，请尝试更换更清晰的照片！")
			return []

		# 判断是否是左侧车牌边缘
		if wave_peaks and wave_peaks[0][1] - wave_peaks[0][0] < (wave_peaks[-1][1] - wave_peaks[-1][0]) / 3 and wave_peaks[0][0] == 0:
			wave_peaks.pop(0)

		# 组合分离汉字（假设第一个字符是汉字）
		if len(wave_peaks) > 2:
			start = wave_peaks[0][0]
			for i, wave in enumerate(wave_peaks):
				if wave[1] - start > (wave_peaks[-1][1] - wave_peaks[-1][0]) * 0.7:
					break
				else:
					continue
			if i > 0:
				wave = (wave_peaks[0][0], wave_peaks[i][1])
				wave_peaks = wave_peaks[i + 1:]
				wave_peaks.insert(0, wave)

		lenlst = []
		for i in wave_peaks:
			lenlst.append(i[1]-i[0])

		wave_peaks_sorted = sorted(lenlst)
		# print(wave_peaks_sorted)
		width_max = wave_peaks_sorted[-1]
		# print(width_max)

		# 较小的峰去掉
		for i in wave_peaks:
			if i[1]-i[0] < width_max*0.3:
				wave_peaks.remove(i)

		if len(wave_peaks) <= 5:
			# print("无法进行字符分割，请尝试更换更清晰的照片！")
			return []

		part_cards = [gray_img[:, wave[0]:wave[1]] for wave in wave_peaks]

		# 可视化分割结果（可选）
		# for idx, part_card in enumerate(part_cards):
		# 	cv2.imwrite(f'./character_{idx}.jpg', part_card)

		return part_cards
	
# 用法示例 README!!!
if __name__ == '__main__':
	# 创建对象
	locator = PlatesLocator()
	# 获取车牌图像列表和对应的车牌颜色列表
	original_img, plate_imgs, plate_colors = locator.locate_plates("./Location/dataset/Green/4.jpg", para_type="ORIGIN")
	# plate_imgs, plate_colors = locator.locate_plates("camera")
	# 摄像头出现问题
	if type(plate_imgs) == type(0) and type(plate_colors) == type(0):
		print("请检查摄像头！")
	# 未成功裁剪车牌
	elif plate_imgs == [] or plate_colors == []:
		print("未检测到车牌，请检查输入图片或尝试更换更清晰的照片！")
	else:
		for index, (plate_img, plate_color) in enumerate(zip(plate_imgs, plate_colors)):
			print(f"车牌颜色：{plate_color}")
			if plate_img is not None and plate_img.size > 0:
				# 获取字符列表
				characters = locator.separate_characters(plate_img, color=plate_color) 
				cv2.imwrite(f"./dataset/Plates/morphologyr/32.jpg", plate_img)
				# cv2.imshow(f"plate_{index}", plate_img)
		
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()