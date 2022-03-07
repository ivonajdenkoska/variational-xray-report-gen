import torch.nn as nn

from utils import *


class Prior(nn.Module):
    def __init__(self):
        super(Prior, self).__init__()
        self.latent_size = LATENT_SIZE
        self.hidden_size = HIDDEN_CVAE_SIZE

        self.hidden_to_latent = nn.Linear(self.hidden_size, self.latent_size)
        self.linear_mu = nn.Linear(self.latent_size, self.latent_size)
        self.linear_var = nn.Linear(self.latent_size, self.latent_size)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=DROPOUT)

        self.init_weights()

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'batch' not in name \
                    and value.requires_grad:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def forward(self, img_token):
        h_z = self.relu(self.dropout(self.hidden_to_latent(img_token)))

        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var


class ApproximatePosterior(nn.Module):
    def __init__(self):
        super(ApproximatePosterior, self).__init__()
        self.latent_size = LATENT_SIZE
        self.hidden_size = HIDDEN_CVAE_SIZE

        self.hidden_to_latent = nn.Linear(self.hidden_size, self.latent_size)
        self.linear_mu = nn.Linear(self.latent_size, self.latent_size)
        self.linear_var = nn.Linear(self.latent_size, self.latent_size)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=DROPOUT)

        self.init_weights()

    def init_weights(self):
        for name, value in self.named_parameters():
            if 'norm' not in name and 'batch' not in name \
                    and value.requires_grad:
                if 'bias' in name:
                    nn.init.zeros_(value)
                elif 'weight' in name:
                    nn.init.kaiming_uniform_(value)

    def forward(self, semantic_features):
        h_z = self.relu(self.dropout(self.hidden_to_latent(semantic_features)))

        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var


def reparametrize(mu, log_var):
    device = set_device()
    # random numbers from a normal distribution with mean 0 and variance 1
    eps = torch.randn(mu.shape[0], mu.shape[1]).to(device)
    return mu + eps * torch.exp(0.5 * log_var)


