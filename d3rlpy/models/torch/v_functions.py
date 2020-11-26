import torch.nn as nn
import torch.nn.functional as F


def create_value_function(observation_shape, encoder_factory):
    encoder = encoder_factory.create(observation_shape)
    return ValueFunction(encoder)


class ValueFunction(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc(h)

    def compute_error(self, obs_t, ret_t):
        v_t = self.forward(obs_t)
        loss = F.mse_loss(v_t, ret_t)
        return loss
