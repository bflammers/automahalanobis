
import torch
import torch.utils.data as data_utils
import numpy as np

import h5py
from scipy.io import loadmat


class Scaler:

    def __init__(self, x):
        # Numpy array input to tensor
        x = torch.from_numpy(x).double()

        # Calculate mean and standard deviation of train
        self.mean_vec = torch.mean(x, dim=0)
        self.sd_vec = torch.std(x, dim=0)

    def to(self, device):
        self.mean_vec = self.mean_vec.to(device)
        self.sd_vec = self.sd_vec.to(device)

    def normalize(self, x):
        return (x - self.mean_vec) / self.sd_vec


def np_shuffle_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def read_mat(path: str, transpose=True, print_dim=False):

    # Read data - different .mat versions: first try h5py, then scipy
    try:
        file = h5py.File(path, 'r')
    except OSError:
        file = loadmat(path)

    # Extract X and labels
    X = np.array(file.get('X'))
    labels = np.array(file.get('y'))

    # Transpose data
    if transpose:
        X = X.transpose()
        labels = labels.transpose()

    if print_dim:
        print('Input data dim:')
        print(' X:      {}'.format(X.shape))
        print(' labels: {}'.format(labels.shape))

    return X, labels


def generate_loaders(X, labels, args, **kwargs):

    # Train validation test split
    X, labels = np_shuffle_arrays(X, labels)

    data_nrows = X.shape[0]
    val_size = int(args.val_prop * data_nrows)
    test_size = int(args.test_prop * data_nrows)

    splits = [data_nrows - val_size - test_size, data_nrows - val_size]
    X_train, X_val, X_test = np.split(X, splits)
    labels_train, labels_val, labels_test = np.split(labels, splits)

    # Fit scaler
    scaler = Scaler(X_train)

    # Pytorch data loaders
    train = data_utils.TensorDataset(torch.from_numpy(X_train).double(),
                                     torch.from_numpy(labels_train).double())
    train_loader = data_utils.DataLoader(train,
                                         batch_size=args.batch_size,
                                         shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(X_val).double(),
                                          torch.from_numpy(labels_val).double())
    val_loader = data_utils.DataLoader(validation,
                                       batch_size=args.batch_size,
                                       shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(X_test).double(),
                                    torch.from_numpy(labels_test).double())
    test_loader = data_utils.DataLoader(test,
                                        batch_size=args.batch_size,
                                        shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, scaler


def load_kdd_smtp(args, as_numpy, **kwargs):

    # Set args
    args.layer_dims = (3, 10, 2, 10, 3)

    # Load data
    X, labels = read_mat('./data/kdd_smtp/kdd_smtp.mat',
                         transpose=True, print_dim=True)

    if as_numpy:
        return X, labels

    # Split data and generate the data loaders
    train_loader, val_loader, test_loader, scaler = \
        generate_loaders(X, labels, args, **kwargs)

    return train_loader, val_loader, test_loader, scaler, args


def load_kdd_http(args, as_numpy, **kwargs):

    # Set args
    args.layer_dims = (3, 10, 2, 10, 3)

    # Load data
    X, labels = read_mat('./data/kdd_http/kdd_http.mat',
                         transpose=True, print_dim=True)

    if as_numpy:
        return X, labels

    # Split data and generate the data loaders
    train_loader, val_loader, test_loader, scaler = \
        generate_loaders(X, labels, args, **kwargs)

    return train_loader, val_loader, test_loader, scaler, args


def load_shuttle(args, as_numpy, **kwargs):

    # Set args
    args.layer_dims = (9, 20, 5, 20, 9)

    # Load data
    X, labels = read_mat('./data/shuttle/shuttle.mat',
                         transpose=False, print_dim=True)

    if as_numpy:
        return X, labels

    # Split data and generate the data loaders
    train_loader, val_loader, test_loader, scaler = \
        generate_loaders(X, labels, args, **kwargs)

    return train_loader, val_loader, test_loader, scaler, args


def load_forest_cover(args, as_numpy, **kwargs):

    # Set args
    args.layer_dims = (10, 20, 5, 20, 10)

    # Load data
    X, labels = read_mat('./data/forest_cover/forest_cover.mat',
                         transpose=False, print_dim=True)

    if as_numpy:
        return X, labels

    # Split data and generate the data loaders
    train_loader, val_loader, test_loader, scaler = \
        generate_loaders(X, labels, args, **kwargs)

    return train_loader, val_loader, test_loader, scaler, args


def load_dataset(args, **kwargs):
    '''
    Load torch data loaders for datasets: kdd_smtp, kdd_http

    :param args: Namespace object created by argparse containing:
        dataset_name, test_prop, val_prop, batch_size
    :param kwargs: to be passed to torch.utils.data.DataLoader
    :return: Tuple: train_loader, val_loader, test_loader, labels_split, args
    '''
    if args.dataset_name == 'kdd_smtp':
        data_tuple = load_kdd_smtp(args, False, **kwargs)
    elif args.dataset_name == 'kdd_http':
        data_tuple = load_kdd_http(args, False, **kwargs)
    elif args.dataset_name == 'shuttle':
        data_tuple = load_shuttle(args, False, **kwargs)
    elif args.dataset_name == 'forest_cover':
        data_tuple = load_forest_cover(args, False, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')
    return data_tuple


if __name__ == "__main__":

    X_train = np.random.randn(20, 5)
    scaler = Scaler(X_train)
    X_scaled = scaler.normalize(X_train)

    np.testing.assert_almost_equal(np.array([0,0,0,0,0]),
                                   np.mean(X_scaled, axis=0))
    np.testing.assert_almost_equal(np.array([1, 1, 1, 1, 1]),
                                   np.std(X_scaled, axis=0))

    from argparse import Namespace
    data_args = Namespace(dataset_name='forest_cover',
                          test_prop=0.2,
                          val_prop=0.2,
                          batch_size=128)

    train_loader, val_loader, test_loader, scaler, args= \
        load_dataset(args=data_args)
