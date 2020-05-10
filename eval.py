import os
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import fer2013_dataset
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
from eval import validate
import matplotlib.pyplot as plt

def validate_loss(loader, model, device):
    epoch_loss = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_corrects = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
    return epoch_loss / len(loader), num_corrects.double() / (len(loader) * imgs.size(0))