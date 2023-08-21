#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torch.optim as optim
import torch.nn as nn
import time
from torch.optim import lr_scheduler
from datetime import date, datetime
from torch.utils.data import random_split

from torchvision import transforms, models
import MyEDFImports as m
from tempfile import TemporaryDirectory

from Transfer_learning.my_pytorch_imports import DatasetFromNp

# In[2]:
data_dir = '/home/tadeusz/Desktop/Tadeusz/mgr_sleep_states/images_cwt_ssq/(15017, 224, 224)_my_scales_sqpy.npy'
data_np = np.load(data_dir)
d = m.load_all_data()
targets = m.load_all_labels()
d, targets = m.remove_ecg_artifacts(d, targets)
targets = m.two_stages_transform(targets)
assert len(targets) == len(data_np)

# In[3]:
# Hyperparameters
nr_of_classes = len(set(targets))
num_epochs = 25
batch_size = 10
SGD_learning_rate = 0.001
SGD_momentum = 0.9
sched_step_size = 7
sched_gamma = 0.1

from collections import Counter
Counter(targets)

# In[9]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_18.fc.in_features
model_18

# In[21]:

model_18.fc = nn.Linear(num_ftrs, nr_of_classes)
model_18.to(device)
# inbalanced dataset 4 to 1 so adding weights to criterion
crit_weights = torch.tensor([5.8, 1.]).to(device)
#crit_weights = torch.tensor([4., 1., 4.]).to(device)
criterion = nn.CrossEntropyLoss(weight=crit_weights)
optimizer_18 = optim.SGD(model_18.parameters(), lr=SGD_learning_rate, momentum=SGD_momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_18, step_size=sched_step_size, gamma=sched_gamma)
# In[4]:

# setting up tranforms for the images (just normalizing pretty much)
data_transforms = transforms.Compose([
    # to tensor can probably be just torch.from_numpy
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset_all = DatasetFromNp(data_np, target=targets, transform=data_transforms)

generator = torch.Generator().manual_seed(23)
train_data, test_data = random_split(dataset_all, [0.8, 0.2], generator=generator)
datasets = {'train': train_data, 'val': test_data}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
               ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
# remove a fixed generator for training


# In[6]:
# not necessarily test_data subset has to be there can be anything else
inputs, classes = next(iter(dataloaders['train']))
grid = torchvision.utils.make_grid(inputs)


# imshow(grid, title= classes)


# In[7]:
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        val_acc = []
        val_losses = []
        train_acc = []
        train_losses = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)


            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # No idea why now for inputs why I need to transfer it to a float from a double
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # saving the training statistics
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_acc.append(epoch_acc)
                elif phase == 'val':
                    val_losses.append(epoch_loss)
                    val_acc.append(epoch_acc)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, train_acc, val_acc, train_losses, val_losses


# In[22]:


model_18, train_acc, val_acc, *_ = train_model(model_18, criterion, optimizer_18, exp_lr_scheduler,
                                               num_epochs=num_epochs)

# In[11]:


dir_saved_models = '../saved_models'
torch.save(model_18, dir_saved_models + f'/{type(model_18).__name__}_{datetime.now().strftime("%b-%d-%Y-%H-%M")}')

# In[12]:

model_18.eval()
dl_for_all = torch.utils.data.DataLoader(dataset_all, batch_size=1, num_workers=4)
predictions = []
labels = []
with torch.no_grad():
    for i, l in dl_for_all:
        i = i.to(device, dtype=torch.float32)
        l.to(device)
        all_outputs = model_18(i)
        predictions.append(all_outputs)
        labels.append(l)

# In[11]:
predictions = [torch.max(y, 1).indices for y in predictions]

# In[11]:


print(torch.all(torch.tensor(predictions) == 0))

# In[6]:
predictions = torch.flatten(torch.stack(predictions))
labels = torch.flatten(torch.stack(labels))
predictions

# In[21]:
from sklearn.metrics import confusion_matrix

confusion_matrix(labels, predictions.to('cpu'))

# In[22]:

running_corrects = torch.sum(predictions.to('cpu') == labels)
running_corrects / len(labels)

# In[24]:

model_18.eval()
dl_test = torch.utils.data.DataLoader(datasets['val'], batch_size=1, shuffle=True, num_workers=4)
predictions = []
labels = []
with torch.no_grad():
    for i, l in dl_test:
        i = i.to(device, dtype=torch.float32)
        l.to(device)
        all_outputs = model_18(i)
        predictions.append(all_outputs)
        labels.append(l)
predictions = [torch.max(y, 1).indices for y in predictions]
predictions = torch.flatten(torch.stack(predictions)).to('cpu')
labels = torch.flatten(torch.stack(labels))
conf_m = confusion_matrix(labels, predictions.to('cpu'))
conf_m
# In[22]:
train_acc = torch.tensor(train_acc).tolist()
val_acc = torch.tensor(val_acc).tolist()
plt.plot(train_acc)
plt.plot(val_acc)
plt.show()
