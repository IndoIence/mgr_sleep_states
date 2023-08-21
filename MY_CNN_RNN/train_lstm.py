#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from model import ECGDataset, ECG_Classifier_LSTM
from torch.utils.data import DataLoader, random_split
from MyEDFImports import load_all_data, load_all_labels, remove_ecg_artifacts, three_stages_transform
import torch
import torch.nn as nn
import os
from tempfile import TemporaryDirectory

# In[2]:


all_unprepared_data = load_all_data()
all_unprepared_labels = load_all_labels()

print(len(all_unprepared_data))
filtered_data, filter_labels = remove_ecg_artifacts(all_unprepared_data, all_unprepared_labels)
print(len(filtered_data))
# going from 6 labels to three Wake, Nrem, REM
filter_labels = three_stages_transform(filter_labels)
# normalization happens in the ECGDataset

# In[3]:


# Hyperparameters
input_size = 64  # Adjust based on the output from your conv layers
hidden_size = 128
num_layers = 2
num_classes = 3  # [Wake, NonREM, REM]
learning_rate = 0.2
batch_size = 10

# In[4]:


stages = ['train', 'val']
dataset_all = ECGDataset(filtered_data, filter_labels)
train_data, test_data = random_split(dataset_all, [0.8, 0.2])
datasets = {'train': train_data, 'val': test_data}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
               stages}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG_Classifier_LSTM(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)  # Move the model to GPU


# In[22]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_name = f'best_model_params_{type(criterion).__name__}_{type(optimizer).__name__}_{num_epochs}.pt'
        best_model_params_path = os.path.join(tempdir, best_model_name)

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

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
                    inputs = inputs.unsqueeze(1).to(device)
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

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        model.load_state_dict(torch.load(best_model_params_path))
    return model


# In[24]:
class NoOpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass


# In[]


# inbalanced dataset 4 to 1 so adding weights to criterion
crit_weitghts = torch.tensor([4., 1., 4.]).to(device)
criterion = nn.CrossEntropyLoss(weight=crit_weitghts)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
no_scheduler = NoOpScheduler(optimizer)
scheduler = no_scheduler
model = ECG_Classifier_LSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,
                            num_layers=num_layers)
model = model.to(device)

# In[25]:


model = train_model(model, criterion, optimizer, scheduler)

# In[13]:

import matplotlib.pyplot as plt
inputs, labels = next(iter(dataloaders['val']))
print(inputs[0])
plt.plot(inputs[0])
plt.show()
# In[21]:


inputs.unsqueeze(1)[0][0]

# Analizing the error:
# In[22]:

