
import torch

from modules.autoencoder import Autoencoder
from argparse import Namespace

from utils.dataloading import load_forest_cover
from utils.dataloading import load_kdd_http
from utils.dataloading import load_kdd_smtp
from utils.dataloading import load_shuttle

data_args = Namespace(dataset_name='forest_cover')

if __name__ == '__main__':





