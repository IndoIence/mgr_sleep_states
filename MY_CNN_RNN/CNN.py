import matplotlib.pyplot
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

#data_dir = '../images/cwt_cmor_scales100.0_128'
data_dir = '../images/cwt_cmor_scales100.0_128_tiff'
# CenterCrop makes sure that all signals have the same size -> extends smaller ones
init_transforms = transforms.Compose([
    #transforms.CenterCrop([100, 100]),
    transforms.ToTensor()
])

dataset = ImageFolder(data_dir, transform=init_transforms)
img, label = dataset[0]

ex_ind = 190
def display_example_img(img, label):
    print(f'Label: {label}')
    plt.imshow(img[0])
    plt.show()


display_example_img(*dataset[ex_ind])
dataset[ex_ind][0].shape

# In[]
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

batch_size = 128
val_size = 2000
train_size = len(dataset) - val_size

train_data, val_data = random_split(dataset, [train_size, val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

# output
# Length of Train Data : 12034
# Length of Validation Data : 2000

# load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)


# In[]
from torchvision.utils import make_grid


def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    plt.show()


show_batch(val_dl)

import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))