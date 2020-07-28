import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, layers, image_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, image_dim, layers):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(image_dim, layers[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(layers[0], layers[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(layers[1], 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
