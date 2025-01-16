
import numpy as np
import cv2
import os
import torch.nn as nn
import torch
import json
import locate
import os


def delete_images_in_folder(folder_path):
    # 定义常见的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            try:
                # 删除文件
                os.remove(file_path)
            except Exception as e:
                print()




#################################################输入模型部分#########################################################
####################################################################################################################

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
###################################################################################################################
class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):

    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()

#########################################################################################################################
def preprocess_image(img_path, img_size):
    """
    Preprocess a single image for prediction.
    """
    # Load the image
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image: {img_path}")

    # Resize and normalize the image
    image = cv2.resize(image, tuple(img_size))
    image = image.astype('float32')
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))  # Convert to channel-first format
    return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension


def predict_single_image(img_path, model, device, img_size):
    """
    Predict the license plate number for a single image.
    """
    # Preprocess the image
    image = preprocess_image(img_path, img_size).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        prebs = model(image)
    prebs = prebs.cpu().numpy()

    # Greedy decode
    preb = prebs[0, :, :]
    preb_label = []
    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))
    no_repeat_blank_label = []
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label:  # Drop repeated and blank labels
        if (pre_c == c) or (c == len(CHARS) - 1):
            pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c

    # Convert label indices to characters
    result = ''.join([CHARS[i] for i in no_repeat_blank_label])
    return result


if __name__ == "__main__":
    folder_path = "output1"
    delete_images_in_folder(folder_path)

    locator = locate.PlatesLocator()
    # 获取车牌图像列表和对应的车牌颜色列表
#############################################################################################################
    ########################这里是整个脚本的输入端口###############################
    plate_imgs, plate_colors = locator.locate_plates("19.jpg")#################在这里输入
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
            if plate_img is not None:
                # 获取字符列表
                characters = locator.separate_characters(plate_img, color=plate_color)
                cv2.imwrite(f"output1/output.jpg", plate_img)


    # Specify image and model paths
    img_path = "output1/output.jpg"  # Replace with the path to your image
    pretrained_model_path = "Final_LPRNet_model.pth"  # Replace with your model path
    img_size = [94, 24]  # Image size expected by the model

    # Check if files exist
    if not os.path.exists(img_path):
        raise FileNotFoundError()
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lprnet.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    lprnet.to(device)
    print("Pretrained model loaded successfully.")

    # Predict the license plate number
    result = predict_single_image(img_path, lprnet, device, img_size)
    print(f"Predicted License Plate Number: {result}")

def recognize_license_plate(input_source, output_folder="output1",
                           pretrained_model_path="./Recognition/Final_LPRNet_model.pth",
                           img_size=(94, 24), para="ORIGIN"):
    """
    识别输入图像或摄像头中的车牌号码。

    参数：
        input_source (str): 输入图像的路径或关键词如 "camera" 使用摄像头。
        output_folder (str): 用于保存输出图像的文件夹。
        pretrained_model_path (str): 预训练LPRNet模型的路径。
        img_size (tuple): 模型预期的图像尺寸。

    返回：
        dict: 包含状态和结果或错误信息的字典。
              例如：
              {
                  "status": "success",
                  "license_plate": ["ABC1234", "XYZ5678"]
              }
              或
              {
                  "status": "error",
                  "message": "错误描述"
              }
    """
    try:
        # 步骤 1：清空输出文件夹
        delete_images_in_folder(output_folder)

        # 步骤 2：初始化 PlatesLocator
        locator = locate.PlatesLocator()

        # 步骤 3：在输入源中定位车牌
        original_imgs, plate_imgs, plate_colors = locator.locate_plates(input_source, para_type=para)

        # 步骤 4：处理车牌检测中的潜在错误
        if isinstance(plate_imgs, int) and isinstance(plate_colors, int):
            return {
                "status": "error",
                "original_image": original_imgs,
                "message": "请检查摄像头！"
            }
        elif not plate_imgs or not plate_colors:
            return {
                "status": "error",
                "original_image": original_imgs,
                "message": "未检测到车牌，请检查输入图片或尝试更换更清晰的照片！"
            }

        # 步骤 5：保存车牌图像并分离字符
        license_plates = []
        for index, (plate_img, plate_color) in enumerate(zip(plate_imgs, plate_colors)):
            if plate_img is not None:
                # 分离字符（可根据需要处理）
                characters = locator.separate_characters(plate_img, color=plate_color)
                output_image_path = os.path.join(output_folder, f"output_{index}.jpg")
                cv2.imwrite(output_image_path, plate_img)
                license_plates.append(output_image_path)

        if not license_plates:
            return {
                "status": "error",
                "original_image": original_imgs,
                "message": "车牌图像保存失败！"
            }

        # 步骤 6：检查模型和图像文件是否存在
        results = []
        for img_path in license_plates:
            if not os.path.exists(img_path):
                return {
                    "status": "error",
                    "original_image": original_imgs,
                    "message": f"图像文件未找到: {img_path}"
                }
            if not os.path.exists(pretrained_model_path):
                return {
                    "status": "error",
                    "original_image": original_imgs,
                    "message": f"预训练模型未找到: {pretrained_model_path}"
                }

            # 步骤 7：加载模型
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
            lprnet.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            lprnet.to(device)
            print("预训练模型加载成功。")

            # 步骤 8：预测车牌号码
            result = predict_single_image(img_path, lprnet, device, img_size)
            results.append(result)

        return {
            "status": "success",
            "original_image": original_imgs,
            "cropped_images": plate_imgs,
            "colors": plate_colors,
            "license_plate": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }