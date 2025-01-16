import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

# 导入模型
print("Load model: ResNet50")
resnet50 = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
print("Load model: alexnet")
alexnet = torch.hub.load("pytorch/vision", "alexnet", pretrained=True)
print("Load model: vgg13")
vgg13 = torch.hub.load("pytorch/vision", "vgg13", pretrained=True)

# 数据预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# 特征提取函数
def extract_resnet50_features(x):
    x = resnet50.conv1(x)
    x = resnet50.bn1(x)
    x = resnet50.relu(x)
    x = resnet50.maxpool(x)
    x = resnet50.layer1(x)
    x = resnet50.layer2(x)
    x = resnet50.layer3(x)
    x = resnet50.layer4(x)
    x = resnet50.avgpool(x)

    return x

def extract_alexnet_features(x):
    x = alexnet.features(x)
    x = alexnet.avgpool(x)

    return x

def extract_vgg13_features(x):
    x = vgg13.features(x)
    x = vgg13.avgpool(x)

    return x

# 特征向量归一化
def normalization(vector):
    vector = vector.ravel()
    return vector / np.linalg.norm(vector)

"""
# 提取图像特征
for i in range(1, 51):
    img_path = "Dataset\\" + str(i) + ".jpg"
    test_image = default_loader(img_path)
    input_image = trans(test_image)
    input_image = torch.unsqueeze(input_image, 0)
    start = time.time()
    image_feature = extract_resnet50_features(input_image)
    image_feature = image_feature.detach().numpy()
    image_feature = normalization(image_feature)
    print(f"Time for extracting features for {str(i)}.jpg: {time.time() - start:.2f}")
    print("Save features for " + str(i) + ".jpg !")
    save_path = "resnet50_features/features" + str(i)
    np.save(save_path, image_feature)

for i in range(1, 51):
    img_path = "Dataset\\" + str(i) + ".jpg"
    test_image = default_loader(img_path)
    input_image = trans(test_image)
    input_image = torch.unsqueeze(input_image, 0)
    start = time.time()
    image_feature = extract_alexnet_features(input_image)
    image_feature = image_feature.detach().numpy()
    image_feature = normalization(image_feature)
    print(f"Time for extracting features for {str(i)}.jpg: {time.time() - start:.2f}")
    print("Save features for " + str(i) + ".jpg !")
    save_path = "alexnet_features/features" + str(i)
    np.save(save_path, image_feature)

for i in range(1, 51):
    img_path = "Dataset\\" + str(i) + ".jpg"
    test_image = default_loader(img_path)
    input_image = trans(test_image)
    input_image = torch.unsqueeze(input_image, 0)
    start = time.time()
    image_feature = extract_vgg13_features(input_image)
    image_feature = image_feature.detach().numpy()
    image_feature = normalization(image_feature)
    print(f"Time for extracting features for {str(i)}.jpg: {time.time() - start:.2f}")
    print("Save features for " + str(i) + ".jpg !")
    save_path = "vgg13_features/features" + str(i)
    np.save(save_path, image_feature)

"""

# 计算所有图片间的相似度
def calculate_similarity(features_path, num_images):
    # 加载所有图片的特征向量
    features = []
    for i in range(1, num_images + 1):
        feature_file = os.path.join(features_path, f"features{i}.npy")
        feature = np.load(feature_file)
        features.append(feature)
    
    features = np.array(features)
    
    # 初始化相似度矩阵
    similarity_matrix = np.zeros((num_images, num_images))
    
    # 归一化特征向量
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # 计算两两图片之间的欧氏距离
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                distance = np.linalg.norm(normalized_features[i] - normalized_features[j])
                similarity_matrix[i, j] = distance
    
    return similarity_matrix

# ----------------------以图搜图任务----------------------

test_image = default_loader('panda.png')
input_image = trans(test_image)
input_image = torch.unsqueeze(input_image, 0)

# resnet50
resnet50_test_feature = extract_resnet50_features(input_image)
resnet50_test_feature = resnet50_test_feature.detach().numpy().ravel()

# alexnet
alexnet_test_feature = extract_alexnet_features(input_image)
alexnet_test_feature = alexnet_test_feature.detach().numpy().ravel()

# vgg13
vgg13_test_feature = extract_vgg13_features(input_image)
vgg13_test_feature = vgg13_test_feature.detach().numpy().ravel()

# 计算test与图片库图片间的相似度
def calculate_similarity(test_feature, features):
    similarities = []
    for feature in features:
        distance = np.linalg.norm(test_feature - feature)
        similarities.append(distance)
    return similarities

# 加载图像库特征
resnet50_features_path = 'resnet50_features'
alexnet_features_path = 'alexnet_features'
vgg13_features_path = 'vgg13_features'
num_images = 50
resnet50_features, alexnet_features, vgg13_features = [], [], []
for i in range(1, num_images + 1):
    # resnet50
    resnet50_feature_file = os.path.join(resnet50_features_path, f"features{i}.npy")
    resnet50_feature = np.load(resnet50_feature_file)
    resnet50_features.append(resnet50_feature)

    # alexnet
    alexnet_feature_file = os.path.join(alexnet_features_path, f"features{i}.npy")
    alexnet_feature = np.load(alexnet_feature_file)
    alexnet_features.append(alexnet_feature)

    # vggg13
    vgg13_feature_file = os.path.join(vgg13_features_path, f"features{i}.npy")
    vgg13_feature = np.load(vgg13_feature_file)
    vgg13_features.append(vgg13_feature)
    
resnet50_features = np.array(resnet50_features)
alexnet_features = np.array(alexnet_features)
vgg13_features = np.array(vgg13_features)

# 计算测试图片与图像库中每张图片的相似度
resnet50_similarities = calculate_similarity(resnet50_test_feature, resnet50_features)
alexnet_similarities = calculate_similarity(alexnet_test_feature, alexnet_features)
vgg13_similarities = calculate_similarity(vgg13_test_feature, vgg13_features)

# 排序并选择Top5
# resnet50
sorted_indices = np.argsort(resnet50_similarities)
resnet50_top5_indices = sorted_indices[:5] 

# alexnet
sorted_indices = np.argsort(alexnet_similarities)
alexnet_top5_indices = sorted_indices[:5] 

# vgg13
sorted_indices = np.argsort(vgg13_similarities)
vgg13_top5_indices = sorted_indices[:5] 


# 打印Top5相似图片及其得分
print("----------restnet50----------")
print("Top 5 similar images and their scores:")
for idx in resnet50_top5_indices:
    score = resnet50_similarities[idx]
    img_name = f"{idx + 1}.jpg"
    print(f"Image {img_name}: Score {score:.2f}")

print("----------alexnet----------")
print("Top 5 similar images and their scores:")
for idx in alexnet_top5_indices:
    score = alexnet_similarities[idx]
    img_name = f"{idx + 1}.jpg"
    print(f"Image {img_name}: Score {score:.2f}")

print("----------vgg13----------")
print("Top 5 similar images and their scores:")
for idx in vgg13_top5_indices:
    score = vgg13_similarities[idx]
    img_name = f"{idx + 1}.jpg"
    print(f"Image {img_name}: Score {score:.2f}")