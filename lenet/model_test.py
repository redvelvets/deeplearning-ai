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

    test_accuracies = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            prediction = model(test_data_x)
            prediction_label = torch.argmax(prediction,dim=1)

            test_accuracies += torch.sum(prediction_label == test_data_y.data)
            test_num += test_data_x.size(0)

        test_accuracies = test_accuracies.double().item()/test_num
        print("测试的准确率为: ", test_accuracies)

if __name__ == '__main__':
    LeNet5 = LeNet5()
    LeNet5.load_state_dict(torch.load('./lenet5.pth'))

    test_dataset_loader = test_dataset_process()
    test_model_process(LeNet5, test_dataset_loader)