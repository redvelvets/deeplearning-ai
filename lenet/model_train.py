import copy
import time

from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

from model import LeNet5

def train_validate_dataset_process():
    train_data = FashionMNIST(root="./data",
                              train=True,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    train_data,validate_data = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    validate_loader = Data.DataLoader(validate_data, batch_size=32, shuffle=True, num_workers=0)
    return train_loader,validate_loader

def train_model_process(model, train_loader, validate_loader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化参数
    best_acc = 0.0


    train_losses = []
    validate_losses = []
    train_accuracies = []
    validate_accuracies = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 训练过程中的损失和梯度/准确度
        train_loss = 0.0
        train_accuracy = 0.0
        validate_loss = 0.0
        validate_accuracy = 0.0

        # 样本数量
        train_num = 0
        validate_num = 0

        for batch_idx, (b_x, b_y) in enumerate(train_loader):
            # 特征和标签放入设备
            data, target = b_x.to(device), b_y.to(device)
            # 设置模型为训练模式
            model.train()

            output = model(data)
            predict_label = torch.argmax(output, dim=1)
            loss = criterion(output, target)

            # 将梯度初始化为0
            optimizer.zero_grad()
            loss.backward()
            # 更新权重参数
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_accuracy += torch.sum(predict_label == target.data)

            train_num += data.size(0)

        for batch_idx, (b_x, b_y) in enumerate(validate_loader):
            data, target = b_x.to(device), b_y.to(device)
            model.eval()
            output = model(data)

            predict_label = torch.argmax(output, dim=1)
            loss = criterion(output, target)
            validate_loss += loss.item() * data.size(0)
            validate_accuracy += torch.sum(predict_label == target.data)
            validate_num += data.size(0)

        train_losses.append(train_loss/train_num)
        train_accuracies.append(train_accuracy.double().item()/train_num)
        validate_losses.append(validate_loss/validate_num)
        validate_accuracies.append(validate_accuracy.double().item()/validate_num)
        print('Epoch {} - Train Loss: {:.4f} | Train Accuracy: {:.4f}'.format(epoch, train_losses[-1], train_accuracies[-1]))
        print('Epoch {} - Validate Loss: {:.4f} | Validate Accuracy: {:.4f}'.format(epoch, validate_losses[-1], validate_accuracies[-1]))

        if validate_accuracies[-1] > best_acc:
            best_acc = validate_accuracies[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_used = time.time() - since
        print('训练和验证耗费时间{:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))

    # 保存最优参数模型
    torch.save(best_model_wts, './lenet5.pth')

    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss": train_losses,
        "validate_loss": validate_losses,
        "train_accuracy": train_accuracies,
        "validate_accuracy": validate_accuracies
    })
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label="train loss")
    plt.plot(train_process['epoch'], train_process['validate_loss'], 'bs-', label="validate loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_accuracy'], 'ro-', label="train accuracy")
    plt.plot(train_process['epoch'], train_process['validate_accuracy'], 'bs-', label="validate accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__ == '__main__':
    LeNet5 = LeNet5()
    train_data, validate_loader = train_validate_dataset_process()
    train_process = train_model_process(LeNet5, train_data, validate_loader,10)
    matplot_acc_loss(train_process)