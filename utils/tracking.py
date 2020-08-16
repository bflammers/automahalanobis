
import datetime
import re
import os
import csv
import json
from tensorboardX import SummaryWriter

class Tracker:

    def __init__(self, args):

        # Make signature of experiment
        time_signature = str(datetime.datetime.now())[:19]
        time_signature = re.sub('[^0-9]', '_', time_signature)
        signature = '{}_{}_{}'.format(time_signature, args.model_name,
                                      args.dataset_name)

        # Set directory to store run
        self.dir = './runs/{}/'.format(signature)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Store settings
        settings_dict = vars(args)

        with open(self.dir + 'settings.json', 'w') as file:
            json.dump(settings_dict, file, sort_keys=True, indent=4)

        # Create csv file for appending stuff during training
        with open(self.dir + 'train_metrics.csv', 'w') as file:
            filewriter = csv.writer(file, delimiter=';')
            filewriter.writerow(['epoch', 'train_loss', 'val_loss',
                                 'top1_percent', 'top5_percent',
                                 'top10_percent', 'top25_percent'])

        # Tensorboard writer
        self.tensorboard=args.tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.dir + 'tensorboard/')
            self.k = 0  # Counter for tensorboard events

    def __del__(self):
        if self.tensorboard:
            self.writer.close()

    def track(self, epoch, train_loss, val_loss, top1_percent=0,
              top5_percent=0, top10_percent=0, top25_percent=0):

        # Collect values in list
        metrics = [epoch, train_loss, val_loss, top1_percent, top5_percent,
                   top10_percent, top25_percent]

        # Append to csv file
        with open(self.dir + 'train_metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(metrics)

        # Write tensorboard events
        if self.tensorboard:
            self.writer.add_scalar('data/train_loss', train_loss, self.k)
            self.writer.add_scalar('data/val_loss', val_loss, self.k)
            self.writer.add_scalar('data/top1_percent', top1_percent, self.k)
            self.writer.add_scalar('data/top5_percent', top5_percent, self.k)
            self.writer.add_scalar('data/top10_percent', top10_percent, self.k)
            self.writer.add_scalar('data/top25_percent', top25_percent, self.k)
            self.k += 1

if __name__=='__main__':

    from argparse import Namespace
    args = Namespace(dataset_name='shuttle',
                     test_prop=0.2,
                     val_prop=0.2,
                     batch_size=128,
                     model_name='autoencoder',
                     tensorboard=True)

    t = Tracker(args)

    t.track(10,0.1,0.11,0.111,0.1111,0.11111)
