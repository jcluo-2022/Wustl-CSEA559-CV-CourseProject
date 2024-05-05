from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import pickle, torch
import numpy as np

class ImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]

def load_train_data(img_size, magnitude, batch_size):
    with open('train_dataset.pkl', 'rb') as f:
        train_data, train_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2,magnitude=magnitude),
    ])
    train_dataset = ImageNetDataset(train_data, train_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    f.close()
    return train_loader

def load_val_data_origin(img_size, batch_size, shuffle=True):
    with open('val_dataset.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ])
    val_dataset = ImageNetDataset(val_data, val_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    f.close()
    return val_loader

def load_val_data(model_name, epsilon, batch_size, shuffle=True):
    with open(f'./FGSD/{model_name}/epsilon_{epsilon}/val_dataset.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)


    # Assume val_data and val_labels are tensors; create a TensorDataset
    val_dataset = Dataset(val_data, val_labels)

    # Create DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True  # Useful for CUDA, if you're using GPU
    )

    return val_loader


from timm.utils import accuracy, AverageMeter
from tqdm.auto import tqdm
from visualize import visualize

import torch, random
import torch.nn as nn


def train(model, loss_fn, optimizer, device, train_loader, scheduler, loss_scaler, update_freq, mixup=None,
          random_erase=None):
    loss_ema = -1
    iterator = tqdm(train_loader, total=int(len(train_loader)))

    update = False

    # deit finetunes in eval mode
    if type(model).__name__ == 'VisionTransformerDistilled':
        model.eval()
    else:
        model.train()

    for i, (x, y) in enumerate(iterator):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if mixup:
            x, y = mixup(x, y)

        if random_erase:
            x = random_erase(x)

        update = True if (i + 1) % update_freq == 0 or i + 1 == len(train_loader) else False
        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = loss_fn(pred, y)
        loss_scaler(loss, optimizer, update_grad=update)

        if update:
            for param in model.parameters():
                param.grad = None
            scheduler.step()

        with torch.no_grad():
            if loss_ema < 0:
                loss_ema = loss.item()
            loss_ema = loss_ema * 0.99 + loss.item() * 0.01

            iterator.set_postfix(train_loss=loss_ema)

    return loss_ema


def validate(model, device, val_loader, epoch, can_visualize=False):
    iterator = tqdm(val_loader, total=int(len(val_loader)))
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    loss_ema = -1

    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(iterator):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = loss_fn(pred, y)

            if loss_ema < 0:
                loss_ema = loss.item()
            loss_ema = loss_ema * 0.99 + loss.item() * 0.01

            iterator.set_postfix(val_loss=loss_ema)

            if can_visualize and random.random() < 0.015:
                visualize(x, pred, y, epoch)

            top1, top5 = accuracy(pred, y, topk=(1, 5))
            acc1_meter.update(top1.item(), y.size(0))
            acc5_meter.update(top5.item(), y.size(0))

    return loss_ema, acc1_meter.avg, acc5_meter.avg