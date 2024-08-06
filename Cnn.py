import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from PIL import Image
%matplotlib inline

print(f"version: {torch.__version__}")
print(f"Is MPS Build? {torch.backends.mps.is_built()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Download Pokemon Dataset
path = '../pokemon_data' 
classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
print(f'Total number of categories: {len(classes)}')

class_counts = {}
for c in classes:
    class_counts[c] = len(os.listdir(os.path.join(path, c)))

num_images = sum(list(class_counts.values()))
print(f'Total number of images in dataset: {num_images}')

fig = plt.figure(figsize=(25, 5))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values())).set_title('Number of images in each class')
plt.xticks(rotation=90)
plt.margins(x=0)
plt.show()

# Replicating validation sets
random_seed = 42
torch.manual_seed(random_seed)

dataset = ImageFolder(path)
validation_size = 4000
training_size = len(dataset) - validation_size
train_ds, val_ds = random_split(dataset,[training_size, validation_size])
len(train_ds), len(val_ds)

image_resize = (128, 128) 

# Crops of size 32x32, horizontal flip default probability of 50%
train_tfms = tt.Compose([tt.Resize(image_resize),
                        tt.RandomCrop(96, padding = 4, padding_mode='reflect'),
                        tt.RandomHorizontalFlip(),
                        tt.ToTensor()])
valid_tfms = tt.Compose([tt.Resize(image_resize), tt.ToTensor()])

train_ds.dataset.transform = train_tfms
val_ds.dataset.transform = valid_tfms


batch_size = 256
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(val_ds, batch_size*2, num_workers=3, pin_memory=True)

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1,2,0))
        break
        
show_batch(train_dl)

# Function to calculate mean and standard deviation
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

# Calculate mean and std for training data
train_mean, train_std = calculate_mean_std(train_dl)

print(f"Mean: {train_mean}")
print(f"Standard Deviation: {train_std}")

# Pick GPU, else CPU
def get_default_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]  # Grab each tensor from the list of tensors
    return data.to(device, non_blocking=True)         # Copies data from cpu to gpu, non_blocking allows parallel execution

class DeviceDataLoader():
   
    # Wrap the dataloader to move data to a device
    def __init__(self, dataloader, device):
        self.dataloder = dataloader
        self.device = device
        
    # Yield a batch of data after moving it to device
    # Returns iterator
    def __iter__(self):
        for batch in self.dataloder:
            yield to_device(batch, self.device)
            
    def __len__(self):
        # Number of batches
        return len(self.dataloder)
    
device = get_default_device()
device

# Update transforms with calculated mean and std
stats = (train_mean.tolist(), train_std.tolist())

image_resize = (128, 128) 

train_tfms = tt.Compose([
    tt.Resize(image_resize),
    tt.RandomCrop(128, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.RandomRotation(degrees=15),
    tt.RandomGrayscale(p=0.1),
    tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    tt.RandomAffine(degrees=345, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    tt.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    tt.ToTensor(),
    tt.Normalize(*stats, inplace=True)
])

valid_tfms = tt.Compose([
    tt.Resize(image_resize),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# Apply the updated transforms
train_ds.dataset.transform = train_tfms
val_ds.dataset.transform = valid_tfms

batch_size = 256

# Re-define data loaders if necessary
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(val_ds, batch_size, num_workers=3, pin_memory=True)

show_batch(train_dl)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    # Calculates loss and accuracy relative to each batch
    # Returns dictionary of loss and accuracy
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'validation_loss': loss.detach(), 'validation_accuracy': acc.detach()}  # Detaches loss and acc from other model info stored from how it was derived
    
    # Receives list of dictionaries containing loss and accuracy
    # Finds the mean loss and the mean accuracy for the entire epoch 
    def validation_epoch_end(self, outputs):
        batch_losses = [x['validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()                                   # converts list of tensors into one tensor and extracts mean
        batch_accs = [x['validation_accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'validation_loss': epoch_loss.item(), 'validation_accuracy': epoch_acc.item()}
    
    # Prints epoch summary of loss and accuracy as taken from validation_epoch_end function above
    # Training loss is new
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5}, train_loss: {:.4f}, validation_loss: {:.4f}, validation_accuracy: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['validation_loss'], result['validation_accuracy']))
        
# Stores 1 for correct, 0 for false
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    total_correct = torch.sum(preds == labels).item()
    total_predictions = len(preds)
    return torch.tensor(total_correct / total_predictions)


# Convolutional Layer
def conv_block(in_channels, out_channels, pool=False, dropout_rate=0.2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),     
              nn.ReLU(inplace=True)]
    if pool:
        # Optional: Every 4 pixels replaced with 1
        layers.append(nn.MaxPool2d(2))
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)                                              # List of layers to pass through

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, dropout_rate=0.2):
        super().__init__()
        
        # in_channels -> 64 channels -> 128 channels
        # When pool=True, image dimensions are halved
        self.conv1 = conv_block(in_channels, 64, pool=False, dropout_rate=dropout_rate)                   # 64 x 96 x 96
        self.conv2 = conv_block(64, 128, pool=True, dropout_rate=dropout_rate)                            # 128 x 48 x 48
        self.res1 = nn.Sequential(conv_block(128, 128, pool=False, dropout_rate=dropout_rate), 
                                  conv_block(128, 128, pool=False, dropout_rate=dropout_rate))  # 2 convolutional blocks without pooling or channel change
        
        self.conv3 = conv_block(128, 256, pool=True, dropout_rate=dropout_rate)                           # 256 x 24 x 24
        self.conv4 = conv_block(256, 512, pool=True, dropout_rate=dropout_rate)                           # 512 x 12 x 12
        self.res2 = nn.Sequential(conv_block(512, 512, pool=False, dropout_rate=dropout_rate), 
                                  conv_block(512, 512, pool=False, dropout_rate=dropout_rate))  # 512 x 12 x 12
        
        self.classifier = nn.Sequential(nn.MaxPool2d(12),                       # 512 x 1 x 1
                                        nn.Flatten(), 
                                        nn.Dropout(dropout_rate),              # 512
                                        nn.Linear(512, num_classes))           # 151 outputs
        
    # Method to set dropout rate
    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        
    def forward(self, xb):
        # input -> 2 conv layers -> res layer + prev output -> 2 conv layers -> res layer + prev output -> Pool -> Flatten -> Linear
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
# Calls the validation step for all batches
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()                          # Turns off nn.Dropout
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Trains model
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                               steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()              # Calculates gradients
            
            # Gradient Clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()             # Gradient descent
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)   # Log epoch info
        history.append(result)           # Append validation loss and accuracy to history
    return history

import gc 
from torch import mps

gc.collect()
mps.empty_cache()

# Move Dataloaders and Model to GPU
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
model = to_device(ResNet9(3, 151), device)
model

# Evaluate Untrained Model, marks start of history
history = [evaluate(model, valid_dl)]
history

# Training Parameters and train model
epochs = 50
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, grad_clip=grad_clip,
                         weight_decay=weight_decay, opt_func=opt_func)

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['validation_loss'] for x in history]
    plt.plot(train_losses, '-b')
    plt.plot(val_losses, '-g')
    plt.ylim(top=6, bottom=0)
    plt.xlim(left=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Losses vs No. of Epochs')
    plt.show()
    
plot_losses(history)

def plot_accuracies(history):
    accuracies = [x['validation_accuracy'] for x in history]
    max_accuracy = 0
    for a in accuracies:
        if (a > max_accuracy):
            max_accuracy = a
            
    print(f"Maximum Accuracy Reached: {max_accuracy}%")
    plt.plot(accuracies, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs No. of Epochs')
    plt.show()
  
plot_accuracies(history)

# Download Pokemon Test Dataset
test_path = '../dataset' 
test_classes = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
print(f'Total number of categories: {len(test_classes)}')

test_class_counts = {}
for c in test_classes:
    test_class_counts[c] = len(os.listdir(os.path.join(test_path, c)))
    
print(f'Total number of images in dataset: {sum(list(test_class_counts.values()))}')

# Convert test images to vectors
test_tfms = tt.Compose([tt.Resize(image_resize), tt.ToTensor()])
test_ds = ImageFolder(test_path, transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size*2, shuffle=True, num_workers=3, pin_memory=True)

# Calculate mean and std for test data
test_mean, test_std = calculate_mean_std(test_dl)

print(f"Mean: {test_mean}")
print(f"Standard Deviation: {test_std}")

test_stats = (test_mean.tolist(), test_std.tolist())

# Normalize images and recreate Dataloader for Test Data
test_tfms = tt.Compose([tt.Resize(image_resize), tt.ToTensor(), tt.Normalize(*stats)])
test_ds = ImageFolder(test_path, transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size*2, shuffle=True, num_workers=3, pin_memory=True)

def predict_image(img, model, dataset):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    print(preds, dataset.classes[preds[0].item()])
    return dataset.classes[preds[0].item()]          

def get_test_accuracy(total_correct, test_dataset):
    test_percent_accuracy = (num_correct / len(test_dataset)) * 100
    print(f"Test Accuracy: {test_percent_accuracy}%")

# Run model on entire test dataset to get overall accuracy
count = 0
num_correct = 0
model.set_dropout_rate(0)
for img, label in test_ds:
    count += 1
    prediction = (predict_image(img, model, test_ds)).lower()
    actual = (test_ds.classes[label]).lower()
    print(actual, prediction)
    if actual == "mrmime":
        actual = "mr_mime"
    if actual == prediction:
        num_correct += 1

get_test_accuracy(num_correct, test_ds)


torch.save(model.state_dict(), 'Model85_state_dict.pth')
state_dict_path = 'Model85_state_dict.pth'

# Load model and put in evaluation mode
model = ResNet9(3, 151)
model.load_state_dict(torch.load('Model85_state_dict.pth', map_location=torch.device('cpu')))

