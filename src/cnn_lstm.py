import torch.nn as nn

class CNN_LSTM_LogMel(nn.Module):
    def __init__(self):
        super(CNN_LSTM_LogMel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 * 2, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), 1, -1)
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        return self.classifier(final_hidden)

class CNN_LSTM_MFCC(nn.Module):
    def __init__(self):
        super(CNN_LSTM_MFCC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.ReLU(), nn.Dropout2d(0.1)
        )
        self.downsample = nn.AdaptiveAvgPool2d((1, 36))
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 * 2, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = self.downsample(x)
        x = x.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]
        return self.classifier(final_hidden)

MODELS = {
    "CNN_LSTM_LogMel": CNN_LSTM_LogMel,
    "CNN_LSTM_MFCC": CNN_LSTM_MFCC,
}