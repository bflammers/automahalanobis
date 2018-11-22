
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset
from utils.tracking import Tracker
from utils.experiment import train_model

parser = argparse.ArgumentParser(description='Automahalanobis experiment')

# Autoencoder args
parser.add_argument('--mahalanobis', type=bool, default=True)
parser.add_argument('--mahalanobis_cov_decay', type=float, default=0.1)
parser.add_argument('--distort_inputs', type=bool, default=False)
parser.add_argument('--distort_targets', type=bool, default=False)

# Dataset args
parser.add_argument('--dataset_name', type=str, default='forest_cover',
                    help='name of the dataset')
parser.add_argument('--test_prop', type=str, default=0.2)
parser.add_argument('--val_prop', type=str, default=0.2)

# Training args
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--mseloss', type=bool, default=False,
                    help='boolean whether to use mse loss (True) or L1 loss')
parser.add_argument('--adam', type=bool, default=True,
                    help='boolean whether to use adam optimizer (True) or SGD with momentum')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--tensorboard', type=bool, default=True)

# Collect args and kwargs
args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available() else False
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Set model name
args.model_name = 'ae'
args.model_name += '-mahalanobis' if args.mahalanobis else '-vanilla'
args.model_name += '-distortinputs' if args.distort_inputs else ''
args.model_name += '-distorttargets' if args.distort_targets else ''

if __name__ == '__main__':

    # Load data
    train_loader, val_loader, test_loader, model_args = \
        load_dataset(args, **kwargs)

    # Construct model and cast to double
    model = Autoencoder(model_args.layer_dims, args.mahalanobis,
                        args.mahalanobis_cov_decay, args.distort_inputs)
    model.double()

    # Determine device and copy model
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model.to(device)

    # Instantiate tracker
    tracker = Tracker(args)

    # Construct loss function
    if args.mseloss:
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.L1Loss()

    # Construct optimizer
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                    nesterov=False)

    # Train the model
    model = train_model(model, criterion, optimizer, train_loader, val_loader,
                        tracker, args, device)

    print("Trained model on device: {}".format(device))
