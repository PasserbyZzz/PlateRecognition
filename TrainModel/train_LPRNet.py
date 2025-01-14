import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from load_data import CHARS,LPRDataLoader
from LPRNet import build_lprnet

# Configuration Class
class Config:
    # Training Parameters
    max_epoch = 15
    img_size = [94, 24]
    train_img_dirs = ["train"]
    test_img_dirs = ["test"]
    dropout_rate = 0.5
    learning_rate = 0.1
    lpr_max_len = 8
    train_batch_size = 128
    test_batch_size = 120
    phase_train = True
    num_workers = 8
    cuda = True
    resume_epoch = 0
    save_interval = 2000
    test_interval = 2000
    momentum = 0.9
    weight_decay = 2e-5
    lr_schedule = [4, 8, 12, 14, 16]
    save_folder = './weights/'
    pretrained_model = ''  # e.g., './weights/Final_LPRNet_model.pth'

    # Plot Parameters
    plot_save_path = './training_curve.png'  # 保存训练曲线的路径

    # Data Augmentation Parameters
    gaussian_blur_prob = 0.15 # 15%概率应用高斯模糊
    rotation_prob = 0.1       # 10%概率应用随机旋转
    rotation_degree = 15    # 最大旋转角度为15度

# Utility Functions
def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = [T_length] * len(lengths)
    target_lengths = list(lengths)
    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    lr = base_lr
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack([img for img in imgs], 0)  # 已经是Tensor，无需转换
    labels = np.concatenate(labels).astype(np.int32)
    lengths = list(lengths)
    return imgs, torch.from_numpy(labels), lengths

# Updated Weights Initialization Function
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # print(f'Initialized {m.__class__.__name__} weights with Kaiming Normal.')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
            # print(f'Initialized {m.__class__.__name__} biases with 0.01.')
    elif isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
            # print(f'Initialized {m.__class__.__name__} weights with 1.0.')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            # print(f'Initialized {m.__class__.__name__} biases with 0.0.')
    # Add more layer types if necessary

# Evaluation Function
def Greedy_Decode_Eval(Net, dataset, config):
    Net.eval()  # Set to evaluation mode
    epoch_size = len(dataset) // config.test_batch_size
    batch_iterator = iter(DataLoader(
        dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    ))

    Tp, Tn_1, Tn_2 = 0, 0, 0
    start_time = time.time()

    for _ in range(epoch_size):
        try:
            images, labels, lengths = next(batch_iterator)
        except StopIteration:
            break  # In case the dataset size is not perfectly divisible

        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length].numpy()
            targets.append(label)
            start += length

        if config.cuda:
            images = images.cuda()

        with torch.no_grad():
            prebs = Net(images).cpu().numpy()

        preb_labels = []
        for preb in prebs:
            preb_label = np.argmax(preb, axis=0)
            no_repeat_blank_label = []
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if np.array_equal(targets[i], label):
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp / (Tp + Tn_1 + Tn_2) if (Tp + Tn_1 + Tn_2) > 0 else 0
    end_time = time.time()
    print(f"[Info] Evaluation Accuracy: {Acc:.4f} [Tp: {Tp}, Tn_1: {Tn_1}, Tn_2: {Tn_2}]")
    print(f"[Info] Evaluation Speed: {(end_time - start_time) / len(dataset):.6f} sec per sample")

    Net.train()  # Switch back to training mode
    return Acc  # 返回准确率以便记录

# Plotting Function
def plot_training_curves(training_losses, training_accuracies, testing_accuracies, config):
    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, 'b-', label='training loss')
    plt.title('training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    # 绘制训练和测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, 'g-', label='training accuracy')
    plt.plot(epochs, testing_accuracies, 'r-', label='validation accuracy')
    plt.title('training and test accuracy ')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.plot_save_path)
    plt.show()
    print(f"训练曲线已保存至 {config.plot_save_path}")

# Training Function
def train(config):
    T_length = 18  # Fixed CTC input length
    epoch = config.resume_epoch

    os.makedirs(config.save_folder, exist_ok=True)

    # Define Data Augmentation Transforms
    data_transforms = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=1)], p=config.gaussian_blur_prob),
        transforms.RandomApply([transforms.RandomRotation(degrees=config.rotation_degree)], p=config.rotation_prob),
        transforms.ToTensor(),  # 转换为Tensor并归一化到[0,1]
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
    ])

    # Build Model
    lprnet = build_lprnet(
        lpr_max_len=config.lpr_max_len,
        phase=config.phase_train,
        class_num=len(CHARS),
        dropout_rate=config.dropout_rate
    )
    device = torch.device("cuda:0" if config.cuda else "cpu")
    lprnet.to(device)
    print("成功构建网络！")

    # Load Pretrained Model or Initialize Weights
    if config.pretrained_model:
        lprnet.load_state_dict(torch.load(config.pretrained_model, map_location=device))
        print("成功加载预训练模型！")
    else:
        lprnet.apply(weights_init)
        print("成功初始化网络权重！")

    # Define Optimizer
    optimizer = optim.RMSprop(
        lprnet.parameters(),
        lr=config.learning_rate,
        alpha=0.9,
        eps=1e-08,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # Load Datasets with Transforms
    train_dataset = LPRDataLoader(
        config.train_img_dirs,
        config.img_size,
        config.lpr_max_len,
        transform=data_transforms  # 传入数据增强转换
    )
    test_dataset = LPRDataLoader(
        config.test_img_dirs,
        config.img_size,
        config.lpr_max_len
        # 测试集不需要数据增强，因此不传入transform
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

    # Initialize lists to store metrics
    training_losses = []
    training_accuracies = []
    testing_accuracies = []

    for epoch in range(config.resume_epoch, config.max_epoch):
        print(f"\n=== 开始第 {epoch + 1} 轮训练 ===")
        batch_iterator = iter(DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        ))
        total_loss = 0  # 用于累积整个轮次的损失
        epoch_size = len(train_dataset) // config.train_batch_size

        for iteration in range(epoch_size):
            global_iteration = epoch * epoch_size + iteration

            # Save Model Checkpoint
            if global_iteration != 0 and global_iteration % config.save_interval == 0:
                save_path = os.path.join(config.save_folder, f'LPRNet_iteration_{global_iteration}.pth')
                torch.save(lprnet.state_dict(), save_path)
                print(f"已保存模型检查点，迭代次数: {global_iteration}")

            # Adjust Learning Rate
            lr = adjust_learning_rate(optimizer, epoch, config.learning_rate, config.lr_schedule)

            start_time = time.time()

            # Load Training Data
            try:
                images, labels, lengths = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(DataLoader(
                    train_dataset,
                    batch_size=config.train_batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    collate_fn=collate_fn
                ))
                images, labels, lengths = next(batch_iterator)

            # Prepare CTC Inputs
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

            if config.cuda:
                images = images.cuda()
                labels = labels.cuda()

            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

            # Forward Pass
            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1).log_softmax(2).requires_grad_()

            # Compute Loss
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if loss.item() == np.inf:
                print("遇到无限损失，跳过此批次。")
                continue
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            total_loss += loss_val

            end_time = time.time()

            # Logging every 20 iterations
            if (iteration + 1) % 20 == 0:
                avg_loss = loss_val / 20  # 最近20次迭代的平均损失
                print(
                    f'轮次: {epoch + 1} || '
                    f'迭代: {iteration + 1}/{epoch_size} || '
                    f'损失: {avg_loss:.4f} || '
                    f'批次时间: {end_time - start_time:.4f} 秒 || '
                    f'学习率: {lr:.8f}'
                )
                # 不记录中间损失，以确保training_losses与epochs长度一致

        # 计算并记录该轮次的平均损失
        avg_epoch_loss = total_loss / epoch_size
        training_losses.append(avg_epoch_loss)

        # End of Epoch: Evaluate Training and Testing Accuracy
        print("\n=== 轮次结束，开始评估 ===")
        train_acc = Greedy_Decode_Eval(lprnet, train_dataset, config)
        test_acc = Greedy_Decode_Eval(lprnet, test_dataset, config)
        training_accuracies.append(train_acc)
        testing_accuracies.append(test_acc)
        print(
            f"第 {epoch + 1} 轮总结: "
            f"训练准确率: {train_acc:.4f}, "
            f"测试准确率: {test_acc:.4f}"
        )

    # After all epochs, perform final evaluation
    print("\n=== 训练完成，进行最终评估 ===")
    final_train_acc = Greedy_Decode_Eval(lprnet, train_dataset, config)
    final_test_acc = Greedy_Decode_Eval(lprnet, test_dataset, config)
    training_accuracies.append(final_train_acc)
    testing_accuracies.append(final_test_acc)

    # Save final model
    final_save_path = os.path.join(config.save_folder, 'Final_LPRNet_model.pth')
    torch.save(lprnet.state_dict(), final_save_path)
    print(f"已保存最终模型至 {final_save_path}")

    # Plot Training Curves
    plot_training_curves(training_losses, training_accuracies, testing_accuracies, config)

# Entry Point
if __name__ == "__main__":
    config = Config()
    train(config)
