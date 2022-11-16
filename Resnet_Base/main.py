import torch
import torch.nn as nn
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

from utils import ImageDataset
from model import ResNet50, train, val, test

# Paths
source_folder = "/data/Chest_XRay/"
train_img = "/data/Chest_XRay/img_train/"
test_img = "/data/Chest_XRay/img_test/"
test_csv = "/data/Chest_XRay/Data_test_sing.csv"
train_csv = "/data/Chest_XRay/Data_train_sing.csv"



def main(args):
    # ------ DATA
    print('==> Preparing data..')

    # Transformers
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Form Customs Datasets
    train_data = ImageDataset(csv=train_csv, train=True, test=False, rootDir=train_img, transform=transform_train)
    val_data = ImageDataset(csv=train_csv, train=False, test=False, rootDir=test_img, transform=transform_test)
    test_data = ImageDataset(csv=test_csv, train=False, test=False, rootDir=test_img, transform=None)

    # data loaders
    batch_size = args.batch_size

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=8)

    # ------ MODEL

    print('==> Building model..')

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # intialize the model
    model = ResNet50(pretrained=args.pretrn).to(device)

    # learning parameters
    lr = args.lr
    epochs = args.epochs

    if args.optimizer == 'adam': optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.optimizer == 'sgd': optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()

    if args.lr_sched == 'cosine': scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if args.lr_sched == 'multi': scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs/2, epochs/4], gamma=0.1)
    if args.lr_sched == 'linear': scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr, total_iters=10)
    if args.lr_sched == 'exp': scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)

    # start the training and validation
    train_loss = []
    valid_loss = []
    f1_list = []
    auc_list = []

    for epoch in range(0,epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(model, train_loader, optimizer, criterion, train_data, device)
        valid_epoch_loss = val(model, val_loader, criterion, val_data, device)
        if args.lr_sched is not 'none': scheduler.step()
        f1, auc = test(model, test_loader, device)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        f1_list.append(f1)
        auc_list.append(auc)

        print(f'Train Loss: {train_epoch_loss:.4f}      Val Loss: {valid_epoch_loss:.4f}')
        print(f'F1:         {f1:.4f}      AUC:      {auc:.4f}')


    # save the trained model to disk
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/home/tasos/ad122/base_lines/' + args.exp_name + '.pth')

    # plot and save the train and validation line graphs
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/home/tasos/ad122/base_lines/' + args.exp_name +'_loss.png')
    plt.show()

    # plot and save the train and validation line graphs
    plt.figure(figsize=(10, 7))
    plt.plot(f1_list, color='orange', label='train loss')
    plt.plot(auc_list, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('AUC and F1')
    plt.legend()
    plt.savefig('/home/tasos/ad122/base_lines/' + args.exp_name +'_testing.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Model Training for Medical Streaming Datasets')
    parser.add_argument('-n','--exp_name', type=str, default='med_baseline')
    parser.add_argument('-e','--epochs', type=int, default=20)
    parser.add_argument('-l', '--lr', type=float, default=0.01)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-p', '--pretrn', action='store_true', default=False)
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-s', '--lr_sched', type=str, default='none')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)