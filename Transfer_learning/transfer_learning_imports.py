import os
import time
from tempfile import TemporaryDirectory

import torch
from matplotlib import pyplot as plt

from torch.utils.data import Dataset

class DatasetFromNp(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        # Extending 1channel image to be put to three channels
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        x = x.expand(3, -1, -1)
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.target)


# this imshow should be fixed doesn't work yet
def imshow(inp, title=None):
    # inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# This needs to have dataloaders and stuff added to parameters if this can be offloaded to a different file
#
# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#
#     # Create a temporary directory to save training checkpoints
#     with TemporaryDirectory() as tempdir:
#         best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
#
#         torch.save(model.state_dict(), best_model_params_path)
#         best_acc = 0.0
#
#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch + 1}/{num_epochs}')
#             print('-' * 10)
#
#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()  # Set model to training mode
#                 else:
#                     model.eval()  # Set model to evaluate mode
#
#                 running_loss = 0.0
#                 running_corrects = 0
#
#                 # Iterate over data.
#                 for inputs, labels in dataloaders[phase]:
#                     # No idea why now for inputs why I need to transfer it to a float from a double
#                     inputs = inputs.to(device, dtype=torch.float)
#                     labels = labels.to(device)
#
#                     # zero the parameter gradients
#                     optimizer.zero_grad()
#
#                     # forward
#                     # track history if only in train
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)
#                         _, preds = torch.max(outputs, 1)
#                         loss = criterion(outputs, labels)
#
#                         # backward + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()
#
#                     # statistics
#                     running_loss += loss.item() * inputs.size(0)
#                     running_corrects += torch.sum(preds == labels.data)
#                 if phase == 'train':
#                     scheduler.step()
#
#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#                 # deep copy the model
#                 if phase == 'val' and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     torch.save(model.state_dict(), best_model_params_path)
#
#             print()
#
#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#         print(f'Best val Acc: {best_acc:4f}')
#
#         # load best model weights
#         model.load_state_dict(torch.load(best_model_params_path))
#     return model


