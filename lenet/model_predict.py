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

def test_dataset_process():
    test_data = FashionMNIST(root="./data",
                              train=False,
                              transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor()]),
                              download=True)
    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    return test_loader

def test_model_process(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classifier = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            prediction = model(test_data_x)
            prediction_label = torch.argmax(prediction,dim=1)
            result = prediction_label.item()
            label = test_data_y.item()
            print(f"预测值为: 下标-{result}/内容-{classifier[result]} --> 标签值为 下标-{label}/内容-{classifier[label]}")


if __name__ == '__main__':
    LeNet5 = LeNet5()
    LeNet5.load_state_dict(torch.load('./lenet5.pth'))

    test_dataset_loader = test_dataset_process()
    test_model_process(LeNet5, test_dataset_loader)