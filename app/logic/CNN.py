import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 22 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input shape: [batch_size, channels, depth, height, width]
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 32, depth/2, height/2, width/2]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 64, depth/4, height/4, width/4]
        x = self.pool(F.relu(self.conv3(x)))  # Output: [batch_size, 128, depth/8, height/8, width/8]

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)

        return x