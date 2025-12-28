from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root="./data",
                          train=True,
                        transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (images, labels) in enumerate(train_loader):
    if step > 0:
        break

batch_x = images.squeeze().numpy()
batch_y = labels.numpy()
class_labels = train_data.classes

plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii,:,:], cmap=plt.cm.gray)
    plt.title(class_labels[batch_y[ii]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)

plt.show()