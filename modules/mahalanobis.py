# -*- coding: utf-8 -*-
"""
Mahalanobis module
--------------------------
"""
import torch
import torch.nn as nn

class MahalanobisLayer(nn.Module):

    def __init__(self, dim, decay = 0.1):
        super(MahalanobisLayer, self).__init__()
        self.S = torch.eye(dim, requires_grad = False)
        self.S_inv = torch.eye(dim, requires_grad = False)
        self.decay = decay

    def forward(self, x, x_fit):
        """
        Calculates Mahalanobis distance between x and x_fit
        """
        delta = x - x_fit
        m = torch.dot(torch.matmul(delta, self.S_inv), delta)
        return torch.sqrt(m)

    def update(self, x, x_fit):
        delta = x - x_fit
        self.S = (1 - self.decay) * self.S + self.decay * torch.cov(delta)
        self.S_inv = torch.inverse(self.S)

if __name__ == "__main__":

    from scipy.spatial import distance
    import numpy as np

    v  = torch.Tensor([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    iv = torch.inverse(v)
    x1 = torch.Tensor([1, 0, 0])
    x2 = torch.Tensor([0, 1, 0])
    x3 = torch.Tensor([0, 2, 0])

    # Mahalanobis distance using scipy
    scipy_12 = distance.mahalanobis(x1.numpy(), x2.numpy(), iv.numpy())
    scipy_23 = distance.mahalanobis(x2.numpy(), x3.numpy(), iv.numpy())
    scipy_13 = distance.mahalanobis(x1.numpy(), x3.numpy(), iv.numpy())

    # Mahalanobis distance pytorch implementation
    mah_layer = MahalanobisLayer(3)
    mah_layer.S_inv = iv

    pytorch_12 = mah_layer.forward(x1, x2)
    pytorch_23 = mah_layer.forward(x2, x3)
    pytorch_13 = mah_layer.forward(x1, x3)

    # Check if almost equal
    np.testing.assert_almost_equal(scipy_12, pytorch_12.numpy())
    np.testing.assert_almost_equal(scipy_23, pytorch_23.numpy())
    np.testing.assert_almost_equal(scipy_13, pytorch_13.numpy())


    mah_layer.update(x1, x3)

