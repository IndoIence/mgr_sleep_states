import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


class ECG_Classifier_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ECG_Classifier_LSTM, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.5)

        # Fully Connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))

        # Swap sequence and feature dimensions
        x = x.transpose(1, 2)

        # Pass through LSTM
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # debuging
        # print(x.shape)

        x, _ = self.lstm(x, (h0, c0))

        # Pass through FC layer
        x = self.fc(x[:, -1, :])
        return x


class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # normalize
        x = (x - torch.mean(x)) / torch.std(x)
        return x, y
