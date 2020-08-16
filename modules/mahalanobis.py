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
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))
        self.decay = decay

    def forward(self, x, x_fit):
        """
        Calculates the squared Mahalanobis distance between x and x_fit
        """

        delta = x - x_fit
        m = torch.mm(torch.mm(delta, self.S_inv), delta.t())
        return torch.diag(m)

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def update(self, X, X_fit):
        delta = X - X_fit
        self.S = (1 - self.decay) * self.S + self.decay * self.cov(delta)
        self.S_inv = torch.pinverse(self.S)

if __name__ == "__main__":

    from scipy.spatial import distance
    import numpy as np

    # Some example data for testing
    v  = torch.Tensor([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    iv = torch.inverse(v)
    X1 = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 2, 0]])
    X2 = torch.Tensor([[0, 1, 0], [0, 2, 0], [0, 2, 0]])

    # Squared Mahalanobis distance using scipy
    scipy_dist_list = [distance.mahalanobis(x1.numpy(), x2.numpy(), iv.numpy()) for x1, x2 in zip(X1, X2)]
    scipy_dist = np.array(scipy_dist_list)**2

    # Mahalanobis distance pytorch implementation
    mah_layer = MahalanobisLayer(3, decay=0.99)
    mah_layer.S_inv = iv

    pytorch_dist = mah_layer(X1, X2)

     # Check if almost equal
    np.testing.assert_almost_equal(scipy_dist, pytorch_dist.numpy())

    # Covariance method
    X = torch.rand(10, 3)
    np_cov_X = np.cov(X.numpy(), rowvar=False)
    pytorch_cov_X = mah_layer.cov(X)

    # Check if almost equal
    np.testing.assert_almost_equal(np_cov_X, pytorch_cov_X.numpy())

    # Update method
    X_fit = torch.rand(10, 3)
    delta = X - X_fit
    np_cov_delta = np.cov(delta.numpy(), rowvar=False)
    pytorch_cov_delta = mah_layer.cov(delta)

    # Check if almost equal after enough updates
    for i in range(20):
        mah_layer.update(X, X_fit)
    np.testing.assert_almost_equal(np_cov_delta, mah_layer.S.numpy())

    # Test if numpy inverse and pytorch pseudo inverse are close
    np.testing.assert_almost_equal(np.linalg.inv(np_cov_delta), mah_layer.S_inv.numpy(), decimal=5)
