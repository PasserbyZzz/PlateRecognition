import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QListWidget, QMessageBox, QScrollArea, QSizePolicy,
    QGroupBox, QProgressBar, QStatusBar
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QSize
from recognize import recognize_license_plate  # 确保 test.py 在同一目录下
import numpy as np

def resource_path(relative_path):
    """
    获取资源文件的绝对路径
    - 如果是打包后的环境，从临时目录加载资源
    - 如果是未打包的开发环境，从当前目录加载资源
    """
    if hasattr(sys, '_MEIPASS'):  # 打包后的临时路径
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class LicensePlateGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车牌识别系统")
        self.setGeometry(100, 100, 1400, 900)
        icon_path = resource_path("./Icon/LOGO.png")
        self.setWindowIcon(QIcon(icon_path))  # 添加窗口图标，请确保有 icon.png 文件

        # 初始化质量参数
        self.quality = "HIGH"  # 默认质量

        # 初始化主窗口和布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # 设置主体部分和侧边栏
        self.setup_main_area()
        self.setup_sidebar()

        # 添加状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_main_area(self):
        """设置主体部分，包括上传和摄像头按钮以及原始图像显示。"""
        self.main_area = QGroupBox("操作区")
        self.main_area.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 3px 0 3px;
            }
        """)
        self.main_area_layout = QVBoxLayout()
        self.main_area.setLayout(self.main_area_layout)

        # 按钮布局
        self.buttons_layout = QHBoxLayout()
        self.main_area_layout.addLayout(self.buttons_layout)

        # 上传按钮
        self.upload_button = QPushButton("📂 上传图片")
        self.upload_button.setIcon(QIcon("upload.png"))  # 请确保有 upload.png 图标文件
        self.upload_button.setIconSize(QSize(24, 24))
        self.upload_button.setFixedHeight(50)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)
        self.buttons_layout.addWidget(self.upload_button)

        # 摄像头按钮
        self.camera_button = QPushButton("📷 使用摄像头")
        self.camera_button.setIcon(QIcon("camera.png"))  # 请确保有 camera.png 图标文件
        self.camera_button.setIconSize(QSize(24, 24))
        self.camera_button.setFixedHeight(50)
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.camera_button.clicked.connect(self.use_camera)
        self.buttons_layout.addWidget(self.camera_button)

        # 原始图像显示
        self.original_image_label = QLabel("原始图像")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(800, 600)
        self.original_image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f0f0f0;
            }
        """)
        self.main_area_layout.addWidget(self.original_image_label)

        self.main_layout.addWidget(self.main_area)

    def setup_sidebar(self):
        """设置侧边栏，包括剪裁图像、车牌颜色、识别号码以及质量按钮。"""
        self.sidebar = QGroupBox("结果展示")
        self.sidebar.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #FF5722;
                border-radius: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 3px 0 3px;
            }
        """)
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        # 剪裁图像部分
        self.cropped_label = QLabel("✂️ 剪裁的车牌图像")
        self.cropped_label.setAlignment(Qt.AlignCenter)
        self.cropped_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.sidebar_layout.addWidget(self.cropped_label)

        self.cropped_scroll = QScrollArea()
        self.cropped_scroll.setWidgetResizable(True)
        self.cropped_container = QWidget()
        self.cropped_layout = QVBoxLayout()
        self.cropped_layout.setAlignment(Qt.AlignTop)
        self.cropped_container.setLayout(self.cropped_layout)
        self.cropped_scroll.setWidget(self.cropped_container)
        self.sidebar_layout.addWidget(self.cropped_scroll)

        # 车牌颜色部分
        self.colors_label = QLabel("🎨 车牌颜色")
        self.colors_label.setAlignment(Qt.AlignCenter)
        self.colors_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.sidebar_layout.addWidget(self.colors_label)

        self.colors_list = QListWidget()
        self.colors_list.setStyleSheet("""
            QListWidget {
                background-color: #fff3e0;
                border: 1px solid #FF9800;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
                font-size: 14px;
            }
        """)
        self.sidebar_layout.addWidget(self.colors_list)

        # 识别号码部分
        self.numbers_label = QLabel("🔢 识别出的号码")
        self.numbers_label.setAlignment(Qt.AlignCenter)
        self.numbers_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.sidebar_layout.addWidget(self.numbers_label)

        self.numbers_list = QListWidget()
        self.numbers_list.setStyleSheet("""
            QListWidget {
                background-color: #e0f7fa;
                border: 1px solid #00BCD4;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 5px;
                font-size: 14px;
            }
        """)
        self.sidebar_layout.addWidget(self.numbers_list)

        # 质量按钮
        self.quality_group = QGroupBox("识别质量")
        self.quality_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #9C27B0;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                font-size: 14px;
            }
        """)
        self.quality_layout = QHBoxLayout()
        self.quality_group.setLayout(self.quality_layout)

        # LOW 质量按钮
        self.quality_low_button = QPushButton("🔴 LOW")
        self.quality_low_button.setIcon(QIcon("low.png"))  # 请确保有 low.png 图标文件
        self.quality_low_button.setIconSize(QSize(20, 20))
        self.quality_low_button.setFixedHeight(40)
        self.quality_low_button.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.quality_low_button.clicked.connect(lambda: self.set_quality("LOW"))
        self.quality_layout.addWidget(self.quality_low_button)

        # HIGH 质量按钮
        self.quality_high_button = QPushButton("🟢 HIGH")
        self.quality_high_button.setIcon(QIcon("high.png"))  # 请确保有 high.png 图标文件
        self.quality_high_button.setIconSize(QSize(20, 20))
        self.quality_high_button.setFixedHeight(40)
        self.quality_high_button.setStyleSheet("""
            QPushButton {
                background-color: #43a047;
                color: white;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #43a047;
            }
        """)
        self.quality_high_button.clicked.connect(lambda: self.set_quality("HIGH"))
        self.quality_layout.addWidget(self.quality_high_button)

        self.sidebar_layout.addWidget(self.quality_group)

        # 添加弹性伸缩，以便按钮位于底部
        self.sidebar_layout.addStretch()

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #3f51b5;
                width: 20px;
            }
        """)
        self.sidebar_layout.addWidget(self.progress_bar)

        self.main_layout.addWidget(self.sidebar)

    def set_quality(self, quality):
        """设置质量参数并重新处理当前输入。"""
        self.quality = quality
        QMessageBox.information(self, "质量设置", f"质量已设置为 {quality}。")
        self.status_bar.showMessage(f"质量已设置为 {quality}", 3000)
        # 如果存在当前输入，重新处理
        if hasattr(self, 'current_input'):
            self.process_input(self.current_input)

    def upload_image(self):
        """处理上传图片按钮点击事件。"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options
        )
        if file_name:
            self.current_input = file_name
            self.process_input(file_name)

    def use_camera(self):
        """处理使用摄像头按钮点击事件。"""
        self.current_input = "camera"
        self.process_input("camera")

    def process_input(self, input_source):
        """调用 recognize_license_plate 函数并更新 UI 显示结果。"""
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 设置为无限循环
        self.status_bar.showMessage("正在处理，请稍候...")

        QApplication.processEvents()  # 更新界面

        try:
            # 调用 recognize_license_plate 函数
            output_path = resource_path("output1")
            model_path = resource_path("./GUI/Final_LPRNet_model.pth")
            result = recognize_license_plate(
                input_source=input_source,
                output_folder=output_path,
                pretrained_model_path=model_path,
                img_size=(94, 24),
                para=self.quality  # 传递质量参数
            )

            if result["status"] == "error":
                if input_source != "camera":
                    original_img_path = input_source
                    pixmap = self.convert_cv_qt_from_path(original_img_path, self.original_image_label.width(), self.original_image_label.height())
                    self.original_image_label.setPixmap(pixmap)
                else:
                    # 对于摄像头输入，假设 original_image 包含最新捕获图像的路径或数组
                    original_imgs = result.get("original_image")
                    if isinstance(original_imgs, np.ndarray):
                        pixmap = self.convert_cv_qt_from_array(original_imgs, self.original_image_label.width(), self.original_image_label.height())
                        self.original_image_label.setPixmap(pixmap)
                    elif isinstance(original_imgs, str) and os.path.exists(original_imgs):
                        pixmap = self.convert_cv_qt_from_path(original_imgs, self.original_image_label.width(), self.original_image_label.height())
                        self.original_image_label.setPixmap(pixmap)
                QMessageBox.critical(self, "错误", result["message"])
                self.status_bar.showMessage("处理失败", 3000)
                return

            # 显示原始图像
            if input_source != "camera":
                original_img_path = input_source
                pixmap = self.convert_cv_qt_from_path(original_img_path, self.original_image_label.width(), self.original_image_label.height())
                self.original_image_label.setPixmap(pixmap)
            else:
                # 对于摄像头输入，假设 original_image 包含最新捕获图像的路径或数组
                original_imgs = result.get("original_image")
                if isinstance(original_imgs, np.ndarray):
                    pixmap = self.convert_cv_qt_from_array(original_imgs, self.original_image_label.width(), self.original_image_label.height())
                    self.original_image_label.setPixmap(pixmap)
                elif isinstance(original_imgs, str) and os.path.exists(original_imgs):
                    pixmap = self.convert_cv_qt_from_path(original_imgs, self.original_image_label.width(), self.original_image_label.height())
                    self.original_image_label.setPixmap(pixmap)

            # 显示剪裁图像
            cropped_images = result.get("cropped_images", [])
            self.display_cropped_images(cropped_images)

            # 显示车牌颜色
            self.display_plate_colors(result.get("colors", []))

            # 显示识别号码
            self.display_recognized_numbers(result.get("license_plate", []))

            self.status_bar.showMessage("处理完成", 3000)

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.status_bar.showMessage("处理出错", 3000)

        finally:
            # 隐藏进度条
            self.progress_bar.setVisible(False)

    def clear_layout(self, layout):
        """清空指定布局中的所有小部件。"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_cropped_images(self, cropped_images):
        """在侧边栏中显示剪裁的车牌图像。"""
        # 清空现有图像
        self.clear_layout(self.cropped_layout)

        for idx, img_array in enumerate(cropped_images):
            if isinstance(img_array, np.ndarray) and img_array.size != 0:
                label = QLabel()
                label.setFixedSize(250, 150)
                label.setStyleSheet("""
                    QLabel {
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        padding: 5px;
                        background-color: #fff;
                    }
                """)
                pixmap = self.convert_cv_qt_from_array(img_array, label.width(), label.height())
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignCenter)
                self.cropped_layout.addWidget(label)
            else:
                print(f"无效的图像数组: {img_array}")

    def display_plate_colors(self, colors):
        """显示检测到的车牌颜色。"""
        self.colors_list.clear()
        for color in colors:
            item = f"• {color}"
            self.colors_list.addItem(item)

    def display_recognized_numbers(self, numbers):
        """显示识别出的车牌号码。"""
        self.numbers_list.clear()
        for number in numbers:
            item = f"• {number}"
            self.numbers_list.addItem(item)

    def convert_cv_qt_from_path(self, img_path, width, height):
        """将文件路径的 OpenCV 图像转换为 QPixmap 以在 QLabel 中显示。"""
        image = cv2.imread(img_path)
        if image is None:
            return QPixmap()
        return self.convert_cv_qt_from_array(image, width, height)

    def convert_cv_qt_from_array(self, cv_img, width, height):
        """将 NumPy 数组的 OpenCV 图像转换为 QPixmap 以在 QLabel 中显示。"""
        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # 调整图像大小，保持纵横比
        pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
        pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap

def main():
    app = QApplication(sys.argv)
    gui = LicensePlateGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
