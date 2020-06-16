import torch
import torch.nn as nn


class PixelHead(nn.Module):
    def __init__(self, observation_shape, use_batch_norm=True):
        super().__init__()
        self.observation_shape = observation_shape
        self.use_batch_norm = use_batch_norm

        n_channels = observation_shape[0]
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm1d(512)

        self.fc = nn.Linear(self._get_linear_input_size(), 512)

    def _get_linear_input_size(self):
        x = torch.rand((1, ) + self.observation_shape)
        with torch.no_grad():
            return self._conv_encode(x).view(1, -1).shape[1]

    def _conv_encode(self, x):
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
        h = self._conv_encode(x)

        h = torch.relu(self.fc(h.view(h.shape[0], -1)))
        if self.use_batch_norm:
            h = self.bn4(h)

        return h

    def feature_size(self):
        return 512


class PixelHeadWithAction(PixelHead):
    def __init__(self, observation_shape, action_size, use_batch_norm=True):
        self.action_size = action_size
        super().__init__(observation_shape, use_batch_norm)

    def _get_linear_input_size(self):
        size = super()._get_linear_input_size()
        return size + self.action_size

    def forward(self, x, action):
        h = self._conv_encode(x)

        # cocat feature and action
        h = torch.cat([h.view(h.shape[0], -1), action], dim=1)
        h = torch.relu(self.fc(h))
        if self.use_batch_norm:
            h = self.bn4(h)

        return h


class VectorHead(nn.Module):
    def __init__(self, observation_shape, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.fc1 = nn.Linear(observation_shape[0], 256)
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
    def __init__(self, observation_shape, action_size, use_batch_norm=True):
        concat_shape = (observation_shape[0] + action_size, )
        super().__init__(concat_shape, use_batch_norm)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        return super().forward(x)
