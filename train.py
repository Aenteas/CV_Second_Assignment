import os
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from eval import validate
from models import Model

def train(loaders, dist, args):
    # use checkpoint model if given
    if args.m is None:
        checkpoint = torch.load(args.checkpoint)
        model_name = checkpoint['name']
        model = Model(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
    else: # else init model
        model_name = args.m
        model = Model(model_name)

    # loss and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dist = torch.FloatTensor(dist).to(device) # no epsilon needs to be added, each category has at least one sample
    if args.wl:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=1 / dist)
    
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    path_to_best_model = ""

    # learning rate
    optimizer = optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-08)
    best_loss = sys.maxsize
    early_stop = False
    # epochs
    iternum = 1
    for epoch in range(args.epoch_num):
        epoch_loss = 0
        num_corrects = 0
        tbar = tqdm(loaders['train'])
        # iterate through images
        for i, (imgs, labels) in enumerate(tbar):
            model.train()
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # current training accuracy of epoch
            epoch_acc = num_corrects.double() / ((i + 1) * args.batch_size)

            tbar.set_description('Epoch: [{}/{}], Epoch_loss: {:.5f}, Epoch_acc: {:.5f}'.format(epoch+1, args.epoch_num, epoch_loss/(i + 1), epoch_acc))
            # early stopping
            if iternum % args.num_iter_to_validate == 0:
                print("Validating model ...")
                if epoch > args.num_iter_to_validate:
                    print('Best validation loss: {}'.format(best_loss))
                val_loss, val_acc = validate(loaders['val'], model, device)
                # if we have the best model so far
                if val_loss < best_loss:
                    best_loss = val_loss
                    path_to_checkpoint = os.path.abspath(os.path.join(args.checkpoint, f'model_{model_name}_epoch_{epoch}.pth'))
                    if path_to_best_model:
                        os.remove(path_to_best_model)
                    path_to_best_model = path_to_checkpoint
                    num_checks = 0
                    state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                    torch.save({'model_state_dict': state_dict, 'model_name': model_name}, path_to_checkpoint)
                else: # else we increase patience, if patience reaches the limit we stop
                    num_checks += 1
                    if num_checks >= args.patience:
                        print("Early stopping ...")
                        early_stop = True
                print('Validation loss: {}\n Validation acc: {}'.format(val_loss, val_acc), 'Number of checks: {}'.format(num_checks))
            if early_stop:
                break
            iternum += 1
    return model
