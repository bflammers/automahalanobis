
import torch
import argparse

from modules.autoencoder import Autoencoder
from utils.dataloading import load_dataset
from utils.tracking import Tracker
from utils.experiment import train_model

parser = argparse.ArgumentParser(description='Automahalanobis experiment')

# Autoencoder args
parser.add_argument('--mahalanobis', dest='mahalanobis', action='store_true')
parser.set_defaults(mahalanobis=False)
parser.add_argument('--mahalanobis_cov_decay', type=float, default=1E-4)
parser.add_argument('--distort_inputs', dest='distort_inputs',
                    action='store_true')
parser.set_defaults(distort_inputs=False)
parser.add_argument('--distort_targets', dest='distort_targets',
                    action='store_true')
parser.set_defaults(distort_targets=False)

# Dataset args
parser.add_argument('--dataset_name', type=str, default='forest_cover',
                    help='name of the dataset')
parser.add_argument('--test_prop', type=str, default=0.2)
parser.add_argument('--val_prop', type=str, default=0.2)

# Training args
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_adam',  dest='adam', action='store_false',
                    help='boolean whether to not use adam optimizer but SGD with momentum')
parser.set_defaults(adam=True)
parser.add_argument('--no_cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=True)
parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false')
parser.set_defaults(tensorboard=True)

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
    train_loader, val_loader, test_loader, scaler, model_args = \
        load_dataset(args, **kwargs)

    # Construct model and cast to double
    model = Autoencoder(model_args.layer_dims, args.mahalanobis,
                        args.mahalanobis_cov_decay, args.distort_inputs)
    model.double()

    # Determine device and copy model and scaler
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model.to(device)
    scaler.to(device)

    # Instantiate tracker
    tracker = Tracker(args)

    # Construct loss function
    criterion = torch.nn.L1Loss()

    # Construct optimizer
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                    nesterov=False)

    # Train the model
    model, epoch = train_model(model, criterion, optimizer, train_loader,
                               val_loader, scaler, tracker, args, device)

    print("Trained model on device: {}".format(device))

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, tracker.dir+'model_state')

    # state = torch.load()
    # model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])


