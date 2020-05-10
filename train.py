import os
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from eval import validate
from models import Model

def train(loaders, args):
    # use checkpoint model if given
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model_name = checkpoint['name']
        model = Model(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_name = args.m
        model = Model(model_name)

    # load checkpoint to model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    path_to_best_model = ""

    # loss and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_num//2], gamma=0.1)
    best_loss = sys.maxsize
    early_stop = False
    for epoch in range(args.epoch_num):
        scheduler.step()
        epoch_loss = 0
        num_corrects = 0
        tbar = tqdm(loaders['train'])
        for i, (imgs, labels) in enumerate(tbar):
            model.train()
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_acc = num_corrects.double() / ((i + 1) * args.batch_size)

            tbar.set_description('Epoch: [{}/{}], Epoch_loss: {:.5f}, Epoch_acc: {:.5f}'.format(epoch+1, args.epoch_num, epoch_loss/(i + 1), epoch_acc))
        if epoch % args.num_epoch_to_validate == 0:
            print("Validating model ...")
            if epoch > args.num_epoch_to_validate:
                print('Best validation loss: {}'.format(best_loss))
            val_loss, val_acc = validate(loaders['val'], model, device)
            if val_loss < best_loss:
                best_loss = val_loss
                path_to_checkpoint = os.path.abspath(os.path.join(pths_path, f'model_{name}_epoch_{epoch}.pth'))
                if path_to_best_model:
                    os.remove(path_to_best_model)
                path_to_best_model = path_to_checkpoint
                num_checks = 0
                state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                torch.save({'model_state_dict': state_dict, 'model_name': model_name}, path_to_checkpoint)
            else:
                num_checks += 1
                if num_checks >= args.patience:
                    print("Early stopping ...")
                    early_stop = True
            print('Validation loss: {}\n Validation acc: {}'.format(val_loss, val_acc), 'Number of checks: {}'.format(num_checks))
        if early_stop:
            break
    return model
