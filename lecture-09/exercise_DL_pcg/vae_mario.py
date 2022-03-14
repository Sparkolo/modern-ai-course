from typing import List

import torch
import torch.nn as nn

Tensor = torch.Tensor


class VAEMario(nn.Module):
    def __init__(
        self,
        z_dim: int = 2,
    ):
        super(VAEMario, self).__init__()
        self.w = 14
        self.h = 14
        self.n_sprites = 11
        self.input_dim = 14 * 14 * 11

        self.z_dim = z_dim or 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(nn.Linear(256, self.z_dim))
        self.fc_var = nn.Sequential(nn.Linear(256, self.z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
        )

        print(self)

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Returns the parameters of q(z|x) = N(mu, sigma).
        """
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes to a level (b, c, h, w)
        """
        res = self.decoder(z)
        res = res.view(-1, self.n_sprites, self.h, self.w)
        res = nn.LogSoftmax(dim=1)(res)
        return res

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Samples from q(z|x) using the reparametrization trick.
        """
        std = torch.exp(0.5 * log_var)
        rvs = torch.randn_like(std)

        return rvs.mul(std).add_(mu)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x.view(-1, self.input_dim))

        # Sample z from p(z|x)
        z = self.reparametrize(mu, log_var)

        # Decode this z
        x_prime = self.decode(z)

        return [x_prime, x, mu, log_var]
