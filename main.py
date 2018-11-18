
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset

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
parser.add_argument('--n_epochs', type=int, default=100)
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
    train_loader, val_loader, test_loader, labels_split, model_args = \
        load_dataset(args, **kwargs)

    # Set hidden layer dimensions
    H1, H2, H3 = 10, 2, 10

    # Construct our model by instantiating the class defined above
    model = Autoencoder(model_args.dim_input, H1, H2, H3, args.mahalanobis,
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

        for X_batch, y_batch in train_loader:

            # Copy data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass: Compute predicted y by passing x to the model
            errors = model(X_batch)

            # Compute and print loss
            loss = criterion(errors, y_batch)
            print('Epoch: {}/{} -- Loss: {}'.format(k, args.n_epochs, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar('data/loss', loss.item(), k)

        if model.mahalanobis_layer:
            with torch.no_grad():
                X_fit = model.reconstruct(X_batch)
                model.mahalanobis.update(X_batch, X_fit)

    print("Trained model on device: {}".format(device))

    print(model.mahalanobis.S)
    print(model.mahalanobis.S_inv)

    # Close tensorboard writer
    writer.close()