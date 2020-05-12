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
from sklearn.metrics import ConfusionMatrixDisplay

def infer(dataset, model, args):
    # indices of images to save
    sample_idxs = rnd.sample(list(range(len(dataset))), args.num_save)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    epoch_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_corrects = 0
    # init confusion matrix of inference
    confusion_matrix = np.zeros((7,7), dtype=np.uint)
    with torch.no_grad():
        for i, (imgs, labels, gray) in enumerate(tqdm(loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels.data)

            confusion_matrix[labels.item(), preds.item()] += 1
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            # get emotion
            emotion = dataset.labels[preds.item()]
            # save images with predictions
            if i in sample_idxs:
                save_img = Image.fromarray(gray.squeeze(0).data.cpu().numpy().astype(np.uint8), mode='L')
                save_img.save(os.path.join(args.o, f'{emotion}_{i}.jpg'))
    # overall test loss and accuracy
    test_loss, test_acc = epoch_loss / len(loader), num_corrects.double() / len(loader)
    print('Test loss: {}\n Test acc: {}'.format(test_loss, test_acc))
    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix, np.array(dataset.labels))
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Path to dataset', default='./data/fer2013.csv')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint model')
    parser.add_argument('-o', type=str, help='path_to_results', default='./outputs')
    parser.add_argument('--num_save', type=int, default=40, help='Number of images to save for inference')

    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.o)):
        os.makedirs(args.o)

    # load checkpoint model
    checkpoint = torch.load(args.checkpoint)
    model_name = checkpoint['model_name']
    model = Model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    infer(fer_2013_dataset(args.d, 'test'), model, args)