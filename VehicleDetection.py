# This is a binary classification problem. (Emergence Vehicle & Non-Emergency Vehicle)

# import libraries
import os
import torch
import pandas as pd
import dataset
import numpy as np
from tqdm import tqdm
from PIL import Image,ImageFilter
import torchvision.models as models
import matplotlib.pyplot as plt
from overrides import overrides
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import Dataset, random_split, DataLoader

# load the data path
train_path = 'images/train.csv'
test_path = 'images/test.csv'

# set classes for our problem
classes  = [
    'emergency',
    'non_emergency'
    ]

# get default device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# checking class balance
train = pd.read_csv('images/train.csv')
print(train.columns)
train['emergency_or_not'].value_counts()

# set class index for train dataset
class0_index = train[train.emergency_or_not == 0].index.values
class1_index = train[train.emergency_or_not == 1].index.values

# Images with wrong labels:
fig, ax = plt.subplots(1, 3, figsize=(8, 8))
ls = [3, 1348, 497]
for index, axs in enumerate(ax.flatten()):
    axs.imshow(Image.open(f"images/{train.iloc[ls[index]]['image_names']}"))
    axs.set_xticks([])
    axs.set_yticks([])

# Instead of changing these three labels, I choose to remove them from the dataset.
idx =  [3,1348,497]
#removing rows from train.csv
print(len(train))
train1 = train.drop(axis= 0 ,index = idx)
train1.to_csv('images/train1.csv',index = False)

# Show images of not emergency cars
fig, ax = plt.subplots(1, 3)
ls = [240, 479, 405]
for index, axs in enumerate(ax.flatten()):
    axs.imshow(Image.open(f"images/{train.iloc[ls[index]]['image_names']}"))
    axs.set_xticks([])
    axs.set_yticks([])

# remove these three items which are not emergency cars
idx =  [240,479,405]
#removing rows from train.csv
print(len(train1))
train2 = train1.drop(axis= 0 ,index = idx)
train2.to_csv('images/train2.csv',index = False)
print(len(train2))

# -----------------------------------------------------------------------------------
# Building the dataset
# creating dataset and data loaders
class VehicleDataset(Dataset):
    def __init__(self, csv_name, folder, transform=None, label=False):
        self.label = label
        self.folder = folder
        print(csv_name)
        self.dataframe = pd.read_csv(self.folder+'/'+csv_name+'.csv')
        self.tms = transform

    def __len__(self):
        return len(self.dataframe)
    @overrides
    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)
    @overrides
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_index = row['image_names']
        image_file = self.folder + '/' + img_index
        image = Image.open(image_file)
        if self.label:
            target = row['emergency_or_not']
            if target == 0:
                encode = torch.FloatTensor([1, 0])
            else:
                encode = torch.FloatTensor([0, 1])
            return self.tms(image), encode
        return self.tms(image)
# ------------------------------------------------------------------------------------

# Transform the dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

train_dataset =  VehicleDataset('train2','images',label = True,transform=transform)

print(len(train_dataset))
print(train_dataset[20][1])

plt.imshow(train_dataset[1][0].permute(1,2,0))

# transform the test dataset
test_dataset = VehicleDataset('test','images',transform=transform)
print(len(test_dataset))
test_dataset[0].shape

# split the train and validate datasets
torch.manual_seed(101)
batch_size = 32
val_pct = 0.2
val_size = int(val_pct * len(train_dataset))
train_size = len(train_dataset) - val_size
print(train_size,val_size)
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
len(train_ds), len(val_ds)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)

train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)

# define showbatch function
def show_batch(batch_size=32, dl=None):
    for image, label in dl:
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xticks([]);
        ax.set_yticks([])

        ax.imshow(make_grid(image.cpu(), nrow=5).permute(1, 2, 0))
        break

# define accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _,truths = torch.max(labels,dim = 1)
    return torch.tensor(torch.sum(preds == truths).item() / len(preds))

# define image classification base function
class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        # _,out = torch.max(out,dim = 1)
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)

        # Generate predictions
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)

        score = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach()}

    # this 2 methods will not change .

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))

# define emergency custom model
class EmergencyCustomModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            # nn.Sigmoid(),
        )

def forward(self, xb):
    return self.network(xb)

custom_model = to_device(EmergencyCustomModel(),device)

# Training and Validation Methods
# defining the training method.
# using weight decay and cyclic lr , gradient clipping
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
    #                                             steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # # # Record & update learning rate
            # lrs.append(get_lr(optimizer))
            # sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def fit(epochs, max_lr, model, train_loader, val_loader,
        weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
# --------------------------------------------------------------------------------------------------
# Use simple fit function. Experimenting with weight decay and gradient clipping
# Keeping everything constant and varying only one parameter. we can observe which parameter will effect the learning
# using
wd = [10,1,1e-1,1e-2,1e-3,1e-4]
epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

hist = {}

for weight_decay in wd:
  print(weight_decay)
  custom_model = to_device(EmergencyCustomModel(),device)
  torch.cuda.empty_cache()
  hist[weight_decay] = fit(epochs,lr,custom_model,train_dl,val_dl,weight_decay,
                  opt_func = opt_func)
# ---------------------------------------------------------------------------------------------------
# define histogram function
def get_df(hist,ls):
    data = {}

    for i in ls:
        train_loss = 0
        val_loss = 0
        val_score = 0
        for j in range(epochs):

            train_loss += hist[i][j]['train_loss']
            val_loss += hist[i][j]['val_loss']
            val_score += hist[i][j]['val_score']

        train_loss /= epochs
        val_loss /= epochs
        val_score /= epochs

        data[i] = [train_loss,val_loss,val_score]

    return pd.DataFrame(data,index = ['train_loss','val_loss','val_score']).T

# final learning with
lr  = 1e-3
epochs = 20

best_wd = 1e-4
gradient_clipping = 0

custom_model = to_device(EmergencyCustomModel(),device)
torch.cuda.empty_cache()
hist = fit(epochs,lr,custom_model,train_dl,val_dl,best_wd,
                gradient_clipping,opt_func = opt_func)

# loss function plots
def plot(hist,epochs = 10):
  train_loss = []
  val_loss = []
  val_score = []
  for i in range(epochs):

      train_loss.append(hist[i]['train_loss'])
      val_loss.append(hist[i]['val_loss'])
      val_score.append(hist[i]['val_score'])

  plt.plot(train_loss,label = 'train_loss')
  plt.plot(val_loss,label = 'val_loss')
  plt.legend()
  plt.title('loss')

  plt.figure()
  plt.plot(val_score,label = 'val_score')
  plt.legend()
  plt.title('accuarcy')

# ResNet50 model
class ResNet50(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet50(pretrained=True)

        feature_in = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(feature_in, 2)

    def forward(self, x):
        return self.pretrained_model(x)

# final Learning with
lr  = 1e-4
epochs = 5
opt_func = torch.optim.Adam
best_wd = 1e-4
gradient_clipping = 0

custom_model = to_device(ResNet50(),device)
torch.cuda.empty_cache()
hist = fit(epochs,lr,custom_model,train_dl,val_dl,best_wd,
                gradient_clipping,opt_func = opt_func)

# save the model
torch.save(custom_model.cpu(), 'models/resnet50.pth')

custom_model = to_device(ResNet50(),device)
torch.cuda.empty_cache()
hist = fit_one_cycle(5, 1e-4, custom_model, train_dl, val_dl,
                  weight_decay=1e-4, grad_clip=0, opt_func=torch.optim.Adam)

# define class Densent169
class Densenet169(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.densenet169(pretrained=True)

        feature_in = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(feature_in, 2)

    def forward(self, x):
        return self.pretrained_model(x)

# final Learning with
lr  = 1e-4
epochs = 5
opt_func = torch.optim.Adam
best_wd = 1e-4
gradient_clipping = 0

custom_model2 = to_device(Densenet169(),device)
torch.cuda.empty_cache()
hist = fit(epochs,lr,custom_model2,train_dl,val_dl,best_wd,
                gradient_clipping,opt_func = opt_func)

# save the model
torch.save(custom_model2.state_dict(), 'models/densenet169_final.pt')

# Loading model
loaded_densenet169 = Densenet169()
loaded_densenet169.load_state_dict(torch.load('densenet169_final.pt'))
loaded_densenet169.eval()
print('loaded')

# Preparation Submission File
preds = []
for test_image in test_dataset:
    test_image = test_image.view(1,3,224,224)
    pred  = loaded_densenet169.forward(test_image)
    _,idx = torch.max(pred,dim = 1)
    idx = idx.numpy()[0]
    preds.append(idx)

# Loading the submission file
subs = pd.read_csv('sample_submission.csv')

subs['emergency_or_not'] = preds
subs.to_csv('densenet169_epochs5.csv',index = False)

preds = []
for image, labels in train_dataset:
    image = image.view(1, 3, 224, 224)

    pred = loaded_densenet169.forward(img)
    _, idx = torch.max(pred, dim=1)
    idx = idx.numpy()[0]
    preds.append(idx)
    print(idx)
    break














