from dataset import fer_2013_dataset
from train import train
from infer import infer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Path to dataset', default='./fer2013.csv')
    parser.add_argument('-m', type=str, help='model_name', choices=['vgg4+0+2', 'vgg4+2+2', 'vgg4+4+2', '2' '3'], default=None)
    parser.add_argument('--num_epoch_to_validate', type=int, default=1)
    parser.add_argument('--bs', type=int, help='batch size', default=4)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--num_workers', type=int, help='number of parallel workers', default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_epoch_to_validate', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint', default='./checkpoints')
    parser.add_argument('-o', type=str, help='path_to_results', default='./outputs')

    args = parser.parse_args()
    if not os.path.exists(os.path.abspath(args.o)):
        os.makedirs(args.o)
    if not os.path.exists(os.path.abspath(args.checkpoint)):
        os.makedirs(args.checkpoint)

    # split dataset
    datasets = {split: fer_2013_dataset(split) for split in ['train', 'val', 'test']}
    # for validation we use batch size 1
    loaders = {'train': data.DataLoader(datasets['train'], batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=True),
               'val': data.DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=1, drop_last=False),
               'test': data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=1, drop_last=False),}

    model = train(loaders, args)
    infer(loaders['test'], model, args)