import torch
import torch.nn as nn

from activations import groupsort


class Generator(torch.nn.Module):
    def __init__(self, in_channels, h_channels, out_channels, kernel_size):
        super().__init__()
        self.act = nn.ReLU(True)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=h_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act,

            nn.ConvTranspose2d(in_channels=h_channels, out_channels=h_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act,

            nn.ConvTranspose2d(in_channels=h_channels, out_channels=h_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act,

            nn.ConvTranspose2d(in_channels=h_channels, out_channels=out_channels, kernel_size=kernel_size))

        # self.output = nn.Tanh()

    def forward(self, x):
        return self.main_module(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, h_channels, out_channels, kernel_size):
        super().__init__()
        self.act = groupsort.GroupSort(2)
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=h_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act,

            nn.Conv2d(in_channels=h_channels, out_channels=h_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act,

            nn.Conv2d(in_channels=h_channels, out_channels=1, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=h_channels),
            self.act)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Linear(in_channels=h_channels, out_channels=1, kernel_size=kernel_size))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
