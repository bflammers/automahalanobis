
import numpy as np
import math
import torch

def train_model(model, criterion, optimizer, train_loader, val_loader, scaler,
                tracker, args, device):

    # Performance metrics and tracking
    val_loss, top1, top5, top10, top25 = \
        validate(val_loader, model, criterion, scaler, device)
    tracker.track(0, 0, val_loss, top1, top5, top10, top25)

    for epoch in range(1, args.n_epochs + 1):

        for X_batch, labels_batch in train_loader:

            # Copy data to device
            X_batch, labels_batch = X_batch.to(device), labels_batch.to(device)

            # Scale X
            X_batch = scaler.normalize(X_batch)

            # Forward pass: Compute predicted y by passing x to the model
            out = model(X_batch)

            # Construct y tensor
            y_batch = torch.zeros_like(out) if model.mahalanobis else X_batch

            # Compute and print loss
            loss = criterion(out, y_batch)
            print('Epoch: {}/{} -- Loss: {}'.format(epoch, args.n_epochs,
                                                    loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model.mahalanobis:
                with torch.no_grad():
                    X_fit = model.reconstruct(X_batch)
                    model.mahalanobis_layer.update(X_batch, X_fit)

        # Performance metrics and tracking
        val_loss, top1, top5, top10, top25 = \
            validate(val_loader, model, criterion, scaler, device)
        tracker.track(epoch, loss, val_loss, top1, top5, top10, top25)

    return model, epoch

def outlier_factor(x, x_val):
    err = x - x_val
    err = torch.pow(err, 2)
    err = torch.sum(err, 1)
    return err / len(err)


def performance(anomalies, scores, percentage):

    # Order anomalies (binary vector) by the anomaly score in descending order
    _, ordering = torch.sort(scores, descending=True)
    ordered_anomalies = anomalies[ordering.type(torch.LongTensor)]

    # Number of observations to include in top
    n_top = math.ceil(len(anomalies) * percentage / 100)

    return torch.sum(ordered_anomalies[:n_top]) / torch.sum(anomalies)

def validate(data_loader, model, criterion, scaler, device):

    class FillableArray:

        def __repr__(self):
            return self.X.__str__()

        def __init__(self, n, tensor=False):
            self.n = n
            self.X = torch.Tensor(torch.zeros(n)) if tensor else np.zeros(n)
            self.i = 0

        def fill(self, x):
            stop_ind = self.i + len(x)
            assert self.n >= stop_ind
            self.X[self.i:stop_ind] = x.flatten()
            self.i = stop_ind

    nrow = len(data_loader.dataset)
    anomalies = FillableArray(nrow, tensor=True)
    scores = FillableArray(nrow, tensor=True)
    loss  =0

    for i, (X_val, labels_val) in enumerate(data_loader):

        # Copy to device
        X_val, labels_val = X_val.to(device), labels_val.to(device)

        # Scale X
        X_val = scaler.normalize(X_val)

        # Calculate output of model: reconstructions or Mahalanobis distance
        out = model(X_val)

        # Construct y tensor and calculate loss
        y_val = torch.zeros_like(out) if model.mahalanobis else X_val
        loss = criterion(out, y_val)

        # Determine anomaly scores
        val_scores = out if model.mahalanobis else outlier_factor(out, X_val)

        # Fill anomaly and score tensors to compute performance on full set
        anomalies.fill(labels_val)
        scores.fill(val_scores)

    loss /= i + 1
    top1 = performance(anomalies.X, scores.X, 1).item()
    top5 = performance(anomalies.X, scores.X, 5).item()
    top10 = performance(anomalies.X, scores.X, 10).item()
    top25 = performance(anomalies.X, scores.X, 25).item()

    return loss.item(), top1, top5, top10, top25

if __name__=='__main__':

    x = torch.randn(10,3)
    x_val = torch.randn_like(x)
    print(outlier_factor(x, x_val))

    from utils.dataloading import load_dataset
    from argparse import Namespace
    from modules.autoencoder import Autoencoder

    data_args = Namespace(dataset_name='kdd_smtp',
                          test_prop=0.2,
                          val_prop=0.2,
                          batch_size=128)

    train_loader, val_loader, test_loader, scaler, model_args = \
        load_dataset(args=data_args)

    args = Namespace(mahalanobis=True,
                     mahalanobis_cov_decay=0.9,
                     distort_inputs=False)

    ae = Autoencoder(model_args.layer_dims, args.mahalanobis,
                     args.mahalanobis_cov_decay, args.distort_inputs)
    ae.double()
    device = torch.device("cuda:0" if False else "cpu")
    ae.to(device)

    criterion = torch.nn.L1Loss()
    test = validate(train_loader, ae, criterion, scaler, device)