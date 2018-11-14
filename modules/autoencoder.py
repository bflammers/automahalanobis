# -*- coding: utf-8 -*-
"""
Autoencoder module
--------------------------
"""
import torch
import torch.nn as nn
from modules.mahalanobis import MahalanobisLayer

class Autoencoder(nn.Module):

    def __init__(self, d_in, h1, h2, h3, mahalanobis_layer=False, mahalanobis_cov_decay=0.1):
        super(Autoencoder, self).__init__()

        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(d_in, h1),  # First hidden layer
            nn.Tanh(),            # First hidden layer
            nn.Linear(h1, h2)     # Compression layer (no non-linearity)
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(h2, h3),    # Third hidden layer
            nn.Tanh(),            # Third hidden layer
            nn.Linear(h3, d_in)   # Output layer
        )

        self.mahalanobis_layer = mahalanobis_layer

        if self.mahalanobis_layer:
            self.mahalanobis = MahalanobisLayer(d_in, mahalanobis_cov_decay)

    def forward(self, x):
        x_enc = self.encoding_layers(x)
        x_fit = self.decoding_layers(x_enc)
        if self.mahalanobis_layer:
            return self.mahalanobis(x, x_fit)
        return x_fit

    def encode(self, x):
        return self.encoding_layers(x)

    def decode(self, x):
        return self.decoding_layers(x)

    def reconstruct(self, x):
        x = self.encoding_layers(x)
        x = self.decoding_layers(x)
        return x


if __name__ == "__main__":
    # N is batch size; D_in is input dimension (and thus output dimension);
    # H1, H2 and H3 are hidden layer dimensions
    N, D_in, H1, H2, H3 = 128, 10, 30, 3, 30

    # Create random Tensors to hold inputs and outputs
    x = torch.Tensor(torch.randn(N, D_in))

    # Construct our model by instantiating the class defined above
    model = Autoencoder(D_in, H1, H2, H3, True, 0.001)

    # Select device to train model on and copy model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Also copy data to device, when training on real data this should be done for each mini-batch
    x = x.to(device)

    # Construct our loss function and an optimizer
    #criterion = torch.nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0, nesterov=False)

    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        errors = model(x)

        # Compute and print loss
        loss = criterion(errors, torch.zeros(errors.size(), device=device)) #x_fit.size(), device=device))
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.mahalanobis_layer:
            with torch.no_grad():
                x_fit = model.reconstruct(x)
                model.mahalanobis.update(x, x_fit)

    print("Trained model on device: {}".format(device))

    print(errors)
    print(x)
    print(model.reconstruct(x))
    if model.mahalanobis_layer:
        print(model.mahalanobis.S)
        print(model.mahalanobis.S_inv)
