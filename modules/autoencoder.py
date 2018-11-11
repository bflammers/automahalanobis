# -*- coding: utf-8 -*-
"""
Autoencoder module
--------------------------
"""
import torch

class MahalanobisLayer(torch.nn.Linear):

    def __init__(self, dim):
        raise NotImplementedError

    def distance(self, x, x_fit):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class Autoencoder(torch.nn.Module):

    def __init__(self, d_in, h1, h2, h3):
        super(Autoencoder, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, h1) # First hidden layer
        self.linear2 = torch.nn.Linear(h1, h2)   # Compression layer
        self.linear3 = torch.nn.Linear(h2, h3)   # Third hidden layer
        self.linear4 = torch.nn.Linear(h3, d_in) # Output layer

    def encode(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x

    def decode(self, x):
        x = torch.tanh (self.linear3(x))
        x = self.linear4(x)
        return x

    def forward(self, x):
        encoding = self.encode(x)
        x_fit = self.decode(encoding)

        return x_fit


if __name__ == "__main__":
    # N is batch size; D_in is input dimension (and thus output dimension);
    # H1, H2 and H3 are hidden layer dimensions
    N, D_in, H1, H2, H3 = 64, 20, 100, 10, 100

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)

    # Construct our model by instantiating the class defined above
    model = Autoencoder(D_in, H1, H2, H3)

    # Select device to train model on and copy model to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Also copy data to device, when training on real data this should be done for each mini-batch
    x = x.to(device)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())
    for t in range(100):
        # Forward pass: Compute predicted y by passing x to the model
        x_fit = model(x)

        # Compute and print loss
        loss = criterion(x_fit, x)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Trained model on device: {}".format(device))
