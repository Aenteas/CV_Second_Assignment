import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from torch import nn
import numpy as np
import os
import random as rnd
from models import Model
from dataset import fer_2013_dataset

def infer(dataset, model, args):
    sample_idxs = rnd.sample(list(range(len(dataset))), args.num_save)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    epoch_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_corrects = 0
    num_corrects_per_cat = [0 for _ in dataset.labels]
    num_cats = [0 for _ in dataset.labels]
    for l in dataset.y:
        num_cats[l] += 1
    with torch.no_grad():
        for i, (imgs, labels, gray) in enumerate(tqdm(loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            correct = torch.sum(preds == labels.data)
            num_corrects += correct
            num_corrects_per_cat[preds.item()] += correct
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            emotion = dataset.labels[preds.item()]
            if i in sample_idxs:
                save_img = Image.fromarray(gray.squeeze(0).data.cpu().numpy().astype(np.uint8), mode='L')
                save_img.save(os.path.join(args.o, f'{emotion}_{i}.jpg'))
    # overall test loss and accuracy
    test_loss, test_acc = epoch_loss / len(loader), num_corrects.double() / len(loader)
    print('Test loss: {}\n Test acc: {}'.format(test_loss, test_acc))
    # per category test loss and accuracy
    acc_per_cat = [float(num_correct)/num for num_correct, num in zip(num_corrects_per_cat, num_cats)]
    plt.bar(range(len(dataset.labels)), acc_per_cat, color='rgbc',tick_label=dataset.labels)
    for i,b in enumerate(acc_per_cat):
        plt.text(i, b+0.05, '%.3f' % b, ha='center', va= 'bottom',fontsize=10)  
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Path to dataset', default='./data/fer2013.csv')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('-o', type=str, help='path_to_results', default='./outputs')
    parser.add_argument('--num_save', type=int, default=40, help='Number of images to save for inference')

    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.o)):
        os.makedirs(args.o)

    checkpoint = torch.load(args.checkpoint)
    model_name = checkpoint['model_name']
    model = Model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    infer(fer_2013_dataset(args.d, 'test'), model, args)