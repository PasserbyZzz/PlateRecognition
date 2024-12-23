import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000   # 车牌区域允许最大面积

def point_limit(point):
	'''
	限制点坐标不小于零
	'''
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0
		
class CarLocator:
	'''
	创建CarLocator类, 用于定位车牌
	'''	

	def __init__(self):
		'''
		初始化函数:车牌识别的部分参数保存在js中,便于根据图片分辨率做调整
		'''
		f = open('config.js')
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
		# 绿色车牌：有渐变效果，列中有至少 50% 的像素符合条件
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
				if yh < i:
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
	
	def locate(self, car_pic, resize_rate=1):
		'''
		主方法，完成车牌检测；
		输入->car_pic:输入图片的路径; resize_rate:图像的缩放比例，用于控制输入图像的大小
		输出->card_imgs:检测到的车牌区域的图像列表，每个矩形对应一个可能的车牌区域
		'''
		# Step1: 读取和调整图像
		img = cv2.imread(car_pic)

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
			
		print(f"height: {pic_height}, width: {pic_width}")

		# Step2: 图像预处理
		blur = self.cfg["blur"]
		# 高斯模糊，降低图像噪声，减少干扰
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0) # 图片分辨率调整

		# 转换为灰度图，为后续边缘检测和二值化做准备
		old_img = img
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# 去掉图像中不会是车牌的区域
		kernel = np.ones((20, 20), np.uint8)
		# 开运算，去除图片中不相关的小物体
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
		# 加权叠加，增强车牌区域的对比度
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0) 

		# Step3: 边缘检测
		ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 二值化
		img_edge = cv2.Canny(img_thresh, 100, 200) # Canny边缘检测

		# 使用闭运算和开运算让图像边缘成为一个整体
		kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
		# 闭运算，连接边缘中的小缺口
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		# 开运算，消除边缘区域中的小噪声
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

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
		contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]

		print(f"轮廓数量: {len(contours)}")

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
				box = cv2.boxPoints(rect)
				box = np.intp(box)
				#old_img = cv2.drawContours(old_img, [box], 0, (0, 0, 255), 2)
				#cv2.imshow("edge4", old_img)
				#cv2.waitKey(0)

		print(f"矩形车牌数量: {len(car_contours)}")

		print("精确定位车牌")

		# Step6: 精确定位车牌
		card_imgs = []
		# 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
		for rect in car_contours:
			# 创造角度，使得左、高、右、低拿到正确的值
			if rect[2] > -1 and rect[2] < 1:
				angle = 1
			else:
				angle = rect[2]

			# 扩大范围，避免车牌边缘被排除
			rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle) 

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
				pts2 = np.float32([left_point, heigth_point, new_right_point]) # 字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
				point_limit(new_right_point)
				point_limit(heigth_point)
				point_limit(left_point)
				card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
				card_imgs.append(card_img)
				#cv2.imshow("card", card_img)
				#cv2.waitKey(0)

			# 负角度
			elif left_point[1] > right_point[1]:
				new_left_point = [left_point[0], heigth_point[1]]
				pts2 = np.float32([new_left_point, heigth_point, right_point]) # 字符只是高度需要改变
				pts1 = np.float32([left_point, heigth_point, right_point])
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(old_img, M, (pic_width, pic_height))
				point_limit(right_point)
				point_limit(heigth_point)
				point_limit(new_left_point)
				card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
				card_imgs.append(card_img)
				#cv2.imshow("card", card_img)
				#cv2.waitKey(0)

		# Step7: 确定车牌颜色
		# 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿车牌
		colors = []
		for card_index, card_img in enumerate(card_imgs):
			green = blue = 0
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
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
			color = "no"
			limit1 = limit2 = 0

			if green * 2 >= card_img_count:
				color = "green"
				limit1 = 35
				limit2 = 99
			elif blue * 2 >= card_img_count:
				color = "blue"
				limit1 = 100
				limit2 = 124 # 有的图片有色偏偏紫

			print(f"车牌颜色: {color}")
			colors.append(color)
			#print(blue, green, yello, black, white, card_img_count)
			#cv2.imshow("color", card_img)
			#cv2.waitKey(0)

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

		return plate_imgs, plate_colors

	
# 测试开始
if __name__ == '__main__':
	locator = CarLocator()
	plate_imgs, plate_colors = locator.locate("./dataset/Green/5.jpg")

	for index, (plate_img, plate_color) in enumerate(zip(plate_imgs, plate_colors)):
		print(plate_color)
		if plate_img is not None:
			cv2.imshow(f"plate_{index}", plate_img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()