import torch
import torch.nn as nn


class PixelHead(nn.Module):
    def __init__(self,
                 n_channels=4,
                 linear_input_size=3136,
                 use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(linear_input_size, 512)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm1d(512)

    def conv_encode(self, x):
        h = torch.relu(self.conv1(x))

        if self.use_batch_norm:
            h = self.bn1(h)

        h = torch.relu(self.conv2(h))

        if self.use_batch_norm:
            h = self.bn2(h)

        h = torch.relu(self.conv3(h))

        if self.use_batch_norm:
            h = self.bn3(h)

        return h

    def forward(self, x):
        h = self.conv_encode(x)

        h = torch.relu(self.fc(h))

        if self.use_batch_norm:
            h = self.bn4(h)

        return h

    def feature_size(self):
        return 512


class PixelHeadWithAction(PixelHead):
    def __init__(self,
                 act_size,
                 n_channels=4,
                 linear_input_size=3136,
                 use_batch_norm=True):
        super().__init__(n_channels, linear_input_size + act_size,
                         use_batch_norm)

    def forward(self, x, action):
        h = self.conv_encode(x)

        # cocat feature and action
        h = torch.cat([h, action], dim=1)

        h = torch.relu(self.fc(h))

        if self.use_batch_norm:
            h = self.bn4(h)

        return h


class VectorHead(nn.Module):
    def __init__(self, obs_size, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 256)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        h = torch.relu(self.fc1(x))

        if self.use_batch_norm:
            h = self.bn1(h)

        h = torch.relu(self.fc2(h))

        if self.use_batch_norm:
            h = self.bn2(h)

        return h

    def feature_size(self):
        return 256


class VectorHeadWithAction(VectorHead):
    def __init__(self, obs_size, act_size, use_batch_norm=True):
        super().__init__(obs_size + act_size, use_batch_norm)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=0)
        return super().forward(x)
