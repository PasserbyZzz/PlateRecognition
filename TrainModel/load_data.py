
from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image  # 引入PIL用于图像转换
from torchvision.transforms import transforms

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, transform=None):
        """
        初始化数据加载器

        参数：
            img_dir (list): 包含图像文件的目录列表。
            imgSize (tuple): 目标图像大小 (宽度, 高度)。
            lpr_max_len (int): 车牌标签的最大长度。
            transform (callable, optional): 应用于图像的转换操作。
        """
        self.img_dir = img_dir
        self.img_paths = []

        # 收集所有目录中的图像路径
        for directory in self.img_dir:
            self.img_paths += [el for el in paths.list_images(directory) if
                               el.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(self.img_paths)  # 随机打乱数据
        print(f"Loaded {len(self.img_paths)} images from {img_dir}.")  # 调试信息

        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.transform = transform  # 存储transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        获取指定索引的数据

        参数：
            index (int): 数据索引

        返回：
            tuple: 图像张量，标签列表，标签长度
        """
        filename = self.img_paths[index]
        try:
            # 使用OpenCV读取图像（支持中文路径）
            img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Unable to load image {filename}. Check the file integrity.")

            # 调整图像大小
            height, width, _ = img.shape
            if (height, width) != (self.img_size[1], self.img_size[0]):
                img = cv2.resize(img, self.img_size)

            # 转换颜色空间从BGR到RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            img = Image.fromarray(img)

            # 应用transform（如果提供）
            if self.transform:
                img = self.transform(img)
            else:
                # 如果未提供transform，应用默认预处理
                img = self.default_transform(img)

            # 从文件名中提取标签
            basename = os.path.basename(filename)
            imgname, _ = os.path.splitext(basename)
            imgname = imgname.split("-")[0].split("_")[0]  # 提取标签部分
            label = [CHARS_DICT[c] for c in imgname if c in CHARS_DICT]

            # 验证标签长度是否符合要求
            if len(label) > self.lpr_max_len:
                label = label[:self.lpr_max_len]  # 截断标签

            return img, label, len(label)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            # 跳过出错的文件，递归调用获取下一个数据
            return self.__getitem__((index + 1) % len(self.img_paths))

    def default_transform(self, img):
        """
        默认的图像预处理：转换为Tensor并进行归一化。

        参数：
            img (PIL Image): 输入图像

        返回：
            Tensor: 预处理后的图像张量
        """
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
        ])
        return transform(img)
