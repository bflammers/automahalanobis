
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset
from utils.tracking import Tracker
from utils.experiment import test_performance

parser = argparse.ArgumentParser(description='Automahalanobis experiment')

# Autoencoder args
parser.add_argument('--mahalanobis', type=bool, default=False)
parser.add_argument('--mahalanobis_cov_decay', type=float, default=0.1)
parser.add_argument('--distort_inputs', type=bool, default=False)

# Dataset args
parser.add_argument('--dataset_name', type=str, default='kdd_smtp',
                    help='name of the dataset')
parser.add_argument('--test_prop', type=str, default=0.2)
parser.add_argument('--val_prop', type=str, default=0.2)

# Training args
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--tensorboard', type=bool, default=False)

# Collect args and kwargs
args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available() else False
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

if __name__ == '__main__':

    # Load data
    train_loader, val_loader, test_loader, model_args = \
        load_dataset(args, **kwargs)

    # Construct our model by instantiating the class defined above
    model = Autoencoder(model_args.layer_dims, args.mahalanobis,
                        args.mahalanobis_cov_decay, args.distort_inputs)
    model.double()

    # Select device to train model on and copy model to device
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model.to(device)

    # Construct our loss function and an optimizer
    # criterion = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, nesterov=False)

    for k in range(1, args.n_epochs + 1):

        for X_batch, labels_batch in train_loader:

            # Copy data to device
            X_batch, labels_batch = X_batch.to(device), labels_batch.to(device)

            # Forward pass: Compute predicted y by passing x to the model
            out = model(X_batch)

            # Construct y tensor
            y_batch = torch.zeros_like(out) if model.mahalanobis else X_batch

            # Compute and print loss
            loss = criterion(out, y_batch)
            print('Epoch: {}/{} -- Loss: {}'.format(k, args.n_epochs, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for i X_val, labels_val in val_loader:
            # Track performance
            out_val = model(X_val)
            perf = [test_performance(labels_val, out_val, x) for

            val_loader()

        if args.tensorboard:
            writer.add_scalar('data/loss', loss.item(), k)

        if model.mahalanobis:
            with torch.no_grad():
                X_fit = model.reconstruct(X_batch)
                model.mahalanobis_layer.update(X_batch, X_fit)

    print("Trained model on device: {}".format(device))

    if model.mahalanobis:
        print(model.mahalanobis_layer.S)
        print(model.mahalanobis_layer.S_inv)

    if args.tensorboard:
        # Close tensorboard writer
        writer.close()