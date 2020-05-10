import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def infer(loader, model, args):
    epoch_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_corrects = 0
    num_corrects_per_cat = [0 for _ in loader.labels]
    num_cats = loader.samples.emotion.value_counts()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            emotion = loader.labels[preds.item()]
            num_corrects_per_cat[preds.item()] += num_corrects
            save_img = Image.fromarray(save_img.squeeze(0).permute(1,2,0).data.cpu().numpy().astype(np.uint8), model='L')
            save_img.save(os.path.join(args.o, f'{emotion}_{i}.jpg'))
    # overall test loss and accuracy
    test_loss, test_acc = epoch_loss / len(loader), num_corrects.double() / len(loader)
    print('Test loss: {}\n Test acc: {}'.format(test_loss, test_acc))
    # per category test loss and accuracy
    acc_per_cat = [float(num_correct)/num for num_correct, num in zip(num_corrects_per_cat, num_cats)]
    plt.bar(range(len(loader.labels)), acc_per_cat, color='rgbc',tick_label=loader.labels)
    for i,b in enumerate(acc_per_cat):
        plt.text(i, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=10)  
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Path to dataset', default='./fer2013.csv')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('-o', type=str, help='path_to_results', default='./outputs')

    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.o)):
        os.makedirs(args.o)

    checkpoint = torch.load(args.checkpoint)
    model_name = checkpoint['name']
    model = Model(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    loader = data.DataLoader(fer_2013_dataset(args.d, 'test'), batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    infer(loader, model, args)