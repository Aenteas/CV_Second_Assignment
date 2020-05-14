from dataset import fer_2013_dataset
from train import train
from infer import infer
import argparse
import os
import torch
from torch.utils import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Path to dataset', default='./data/fer2013.csv')
    parser.add_argument('-m', type=str, help='model_name', choices=['vgg2_0_2', 'vgg2_2_2', 'vgg2_4_ilrb_2', 'vgg2_4_2_conv3', 'vgg2_4_2_conv5'], default=None)
    parser.add_argument('--num_iter_to_validate', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--num_workers', type=int, help='number of parallel workers', default=4)
    parser.add_argument('--epoch_num', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--wl', help='Weight loss function with category volumes', action='store_true', default=False)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint', default='./checkpoint/')
    parser.add_argument('--num_save', type=int, default=40, help='Number of images to save for inference')
    parser.add_argument('-o', type=str, help='path_to_results', default='./outputs')

    args = parser.parse_args()
    # create folders if they do not exist
    if not os.path.exists(os.path.abspath(args.o)):
        os.makedirs(args.o)
    if not os.path.exists(os.path.abspath(args.checkpoint)):
        os.makedirs(args.checkpoint)

    # split dataset
    datasets = {split: fer_2013_dataset(args.d, split) for split in ['train', 'val', 'test']}
    # show category distribution
    dist = datasets['train'].cat_distribution()
    # for validation and test we use batch size 1
    loaders = {'train': data.DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True),
               'val': data.DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=1, drop_last=False),
               'test': data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=1, drop_last=False),}

    model = train(loaders, dist, args)
    # run inference on trained model
    infer(datasets['test'], model, args)
