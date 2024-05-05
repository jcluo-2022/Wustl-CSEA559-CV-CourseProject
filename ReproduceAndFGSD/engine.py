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


img_size = 384
batch_size = 4
means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
stds = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
min_vals = ((0 - means) / stds).view(1, 3, 1, 1).expand(batch_size, 3, img_size, img_size).to("cuda:0")
max_vals = ((1 - means) / stds).view(1, 3, 1, 1).expand(batch_size, 3, img_size, img_size).to("cuda:0")


def fgsm_attack(image, gradient, epsilon=4):
    # Get the direction of the gradient
    sign_data_grad = gradient.sign()
    #
    perturbed_image = image + epsilon * sign_data_grad
    # clip the value into valid range
    perturbed_image = torch.max(torch.min(perturbed_image, max_vals), min_vals)
    return perturbed_image


def adversarial_training(model, loss_fn, optimizer, device, train_loader, scheduler, loss_scaler, update_freq,
                         mixup=None,
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
        loss_scaler(loss, optimizer, update_grad=False)

        # compute the gradient of the training image
        x_adv = x.detach().clone()
        x_adv.requires_grad = True
        with torch.cuda.amp.autocast():
            pred_adv = model(x_adv)
            loss_adv = loss_fn(pred_adv, y)
        loss_scaler(loss_adv, optimizer, update_grad=False)

        # Use adversarial image to train
        adv_x = fgsm_attack(image=x_adv, gradient=x_adv.grad)
        with torch.cuda.amp.autocast():
            pred_adv = model(adv_x)
            loss_adv = loss_fn(pred_adv, y)
        loss_scaler(loss_adv, optimizer, update_grad=update)

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
