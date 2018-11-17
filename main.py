
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset

from tensorboardX import SummaryWriter

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Automahalanobis experiment')

# Dataset args
parser.add_argument('--dataset_name', type=str, default='kdd_smtp',
                    help='name of the dataset')
parser.add_argument('--test_prop', type=str, default=0.2)
parser.add_argument('--val_prop', type=str, default=0.2)

# Training args
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=bool, default=True)

# Collect args and kwargs
args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available() else False
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if __name__ == '__main__':

    # Set hidden layer dimensions
    H1, H2, H3 =  10, 2, 10

    # Load data
    train_loader, val_loader, test_loader, labels_split, model_args = \
        load_dataset(args, **kwargs)

    # Construct our model by instantiating the class defined above
    model = Autoencoder(model_args.dim_input, H1, H2, H3, True, 0.1)
    model.double()

    # Select device to train model on and copy model to device
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model.to(device)

    # Also copy data to device, when training on real data this should be done for each mini-batch

    # Construct our loss function and an optimizer
    #criterion = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, nesterov=False)

    i = 0
    for k in range(1, args.n_epochs + 1):

        for X_batch, y_batch in train_loader:

            # Copy data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass: Compute predicted y by passing x to the model
            errors = model(X_batch)

            # Compute and print loss
            loss = criterion(errors, y_batch)
            print('Epoch: {}/{} -- Updatestep: {} -- Loss: {}'.format(k, args.n_epochs, i, loss.item()))

            writer.add_scalar('data/loss', loss.item(), i)
            i += 1

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if model.mahalanobis_layer:
            with torch.no_grad():
                X_fit = model.reconstruct(X_batch)
                model.mahalanobis.update(X_batch, X_fit)

    print("Trained model on device: {}".format(device))

    print(model.mahalanobis.S)
    print(model.mahalanobis.S_inv)

    input('Press key to continue')

    # export scalar data to JSON for external processing
    writer.close()