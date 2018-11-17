
import torch
import torch.utils.data as data_utils
import h5py
import numpy as np

def np_shuffle_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_kdd_smtp(args, **kwargs):

    # Set args
    args.dim_input = 3

    # Read data
    file = h5py.File('data/smtp_kddcup99/smtp.mat', 'r')
    data_X = np.array(file.get('X')).transpose()
    data_labels = np.array(file.get('y')).transpose()

    # Preprocessing

    # Train validation test split
    X, labels = np_shuffle_arrays(data_X, data_labels)

    data_nrows = X.shape[0]
    val_size = int(args.val_prop * data_nrows)
    test_size = int(args.test_prop * data_nrows)

    splits = [data_nrows - val_size - test_size, data_nrows - val_size]
    X_train, X_val, X_test = np.split(X, splits)

    labels_split = dict((x, y) for x, y in zip(('train', 'val', 'test'),
                                               np.split(labels, splits)))

    # y values: arrays containing zeros
    y_train = np.zeros(X_train.shape[0])
    y_val = np.zeros(X_val.shape[0])
    y_test = np.zeros(X_test.shape[0])

    # Pytorch data loaders
    train = data_utils.TensorDataset(torch.from_numpy(X_train),
                                     torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size,
                                         shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(X_val),
                                          torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation,batch_size=args.batch_size,
                                       shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(X_test),
                                    torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size,
                                        shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, labels_split, args


def load_dataset(args, **kwargs):
    '''
    Load torch data loaders for datasets: kdd_smtp, kdd_http

    :param args: Namespace object created by argparse containing:
        dataset_name, test_prop, val_prop, batch_size
    :param kwargs: to be passed to torch.utils.data.DataLoader
    :return: Tuple: train_loader, val_loader, test_loader, labels_split, args
    '''
    if args.dataset_name == 'kdd_smtp':
        data_tuple = load_kdd_smtp(args, **kwargs)
    elif args.dataset_name == 'kdd_http':
        pass
    else:
        raise Exception('Wrong name of the dataset!')
    return data_tuple

if __name__ == "__main__":

    from argparse import Namespace
    data_args = Namespace(dataset_name='kdd_smtp',
                          test_prop=0.2,
                          val_prop=0.2,
                          batch_size=128)

    train_loader, val_loader, test_loader, labels, args = load_dataset(args=data_args)
