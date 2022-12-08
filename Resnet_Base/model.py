import torch
import torch.nn as nn
from torchvision import models as models
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from utils import label_dict

# Make model with 1 channel as input (conv1) and  multi-label (replace the softmax function with a sigmoid)
class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model = resnet
        self.base_model.fc = nn.Linear(in_features=2048, out_features=len(label_dict), bias=True)

    def forward(self, x):
        return self.base_model(x)

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, batch in tqdm(enumerate(dataloader)):
        counter += 1
        data, target = batch
        data = data.to(device)
        target = target.to(device)    

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter

    return train_loss


# validate function
def val(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            counter += 1
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
        
        val_loss = val_running_loss / counter

        return val_loss

def test(model, dataloader, device):
    print('Testing')
    model.eval()
    y_true = []
    y_pred = []
    #print(type(y_pred))

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            data, y_t = batch
            for l in y_t:
                y_true.append(l.tolist())
            y_p = model(data.to(device))
            for l in y_p: 
                l = torch.sigmoid(l)
                y_pred.append(l.tolist())
                #print("After", type(y_pred))
            #y_pred = model(data)

    f1 = f1_score(y_true, y_pred, zero_division=1, average=None)
    auc = roc_auc_score(y_true, y_pred, average=None)
    
    return f1, auc