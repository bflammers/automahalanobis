# -*- coding: utf-8 -*-
"""
Autoencoder module
--------------------------
"""
import torch
import torch.nn as nn
from modules.mahalanobis import MahalanobisLayer

class Autoencoder(nn.Module):

    def __init__(self, layer_dims, mahalanobis=False,
                 mahalanobis_cov_decay=0.1, distort_inputs=False):
        super(Autoencoder, self).__init__()

        self.layer_dims = layer_dims

        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),  # 1st hidden layer
            nn.Tanh(),                                # 1st hidden layer
            nn.Linear(layer_dims[1], layer_dims[2])   # Compression layer
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[2], layer_dims[3]),  # 3rd hidden layer
            nn.Tanh(),                                # 3d hidden layer
            nn.Linear(layer_dims[3], layer_dims[4])   # Output layer
        )

        self.mahalanobis = mahalanobis

        if mahalanobis:
            self.mahalanobis_layer = MahalanobisLayer(layer_dims[0],
                                                      mahalanobis_cov_decay)

        self.distort_input = distort_inputs

    def forward(self, x):
        x_in = x + torch.randn_like(x) if self.distort_input else x
        x_enc = self.encoding_layers(x_in)
        x_fit = self.decoding_layers(x_enc)
        if self.mahalanobis:
            x_fit = self.mahalanobis_layer(x, x_fit)
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
    batch_size = 128
    layer_dims = 10, 30, 5, 30, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.Tensor(torch.randn(batch_size, layer_dims[0]))

    # Construct our model by instantiating the class defined above
    model = Autoencoder(layer_dims, True, 0.001, True)

    # Select device to train model on and copy model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Copy data to device
    x = x.to(device)

    # Construct our loss function and an optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)

    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        errors = model(x)

        # Compute and print loss
        loss = criterion(errors, torch.zeros(errors.size(), device=device))
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model.mahalanobis_layer:
            with torch.no_grad():
                x_fit = model.reconstruct(x)
                model.mahalanobis_layer.update(x, x_fit)

    print("Trained model on device: {}".format(device))

    print(errors)
    print(x)
    print(model.reconstruct(x))
    if model.mahalanobis:
        print(model.mahalanobis_layer.S)
        print(model.mahalanobis_layer.S_inv)
