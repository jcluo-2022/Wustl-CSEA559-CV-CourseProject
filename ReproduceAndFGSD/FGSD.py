import logging

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_aug import get_data_augmentations
from dataset import load_train_data, load_val_data
from engine import train, validate
from fileio import pickle_data
from log import create_logger
from math import ceil
from os import get_terminal_size
from scaler import NativeScaler
from throughput import throughput
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.models import create_model

import argparse, sys, torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from logging import getLogger

from visualize import visualize


def parse_args():
    parser = argparse.ArgumentParser('Vision Transformer training and evaluation script', add_help=False)
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'deit', 'swin', 'cait', 'beit'],
                        help='vit: ViT-L/16, '
                             'deit: DeiT-B/16 distilled, '
                             'swin: Swin-L, '
                             'cait: CaiT-S36')

    parser.add_argument('--model_path', type=str, required=True, help='The path of the model checkpoint')

    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--resume', type=str, help='Resume training')

    # data augmentation
    parser.add_argument('--mixup', action='store_true', default=True)
    parser.add_argument('--no-mixup', action='store_false', dest='mixup', help='Disable mixup')
    parser.add_argument('--cutmix', action='store_true', default=True)
    parser.add_argument('--no-cutmix', action='store_false', dest='cutmix', help='Disable cutmix')
    parser.add_argument('--randerase', action='store_true', default=True)
    parser.add_argument('--no-randerase', action='store_false', dest='randerase', help='Disable random erasing')
    parser.add_argument('--randaug', action='store_true', default=False)

    # optimizer
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--nesterov', action='store_true', default=True, help='Use nesterov momentum for SGD')
    parser.add_argument('--no-nesterov', action='store_false', dest='nesterov', help='Disable nesterov')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay for optimizer')

    parser.add_argument('--label-smooth', type=float, default=0.1, help='Label smoothing percent')

    args, _ = parser.parse_known_args()

    return args


def load_model(model_name, model_path):
    if model_name == 'cait':
        model = create_model('cait_s36_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 128
    elif model_name == 'deit':
        model = create_model('deit_base_distilled_patch16_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    elif model_name == 'swin':
        model = create_model('swin_large_patch4_window12_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 2
    elif model_name == 'vit':
        model = create_model('vit_large_patch16_384', pretrained=True, drop_path_rate=0.1)
        batch_size = 64
    else:
        logger.error('Invalid model name, please use either cait, deit, swin, or vit')
        sys.exit(1)

    model.reset_classifier(num_classes=200)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, batch_size


def load_optimizer(args, model):
    if args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9,
                              nesterov=args.nesterov)
    else:
        logger.error('Invalid optimizer name, please use either adamw or sgd')
        sys.exit(1)

    return optimizer


if __name__ == '__main__':
    args = parse_args()

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, batch_size = load_model(args.model, args.model_path)
    model = model.to(device)

    img_size = 384

    randaug_magnitude = 9 if args.randaug else 0
    # train_loader = load_train_data(img_size, randaug_magnitude, batch_size)
    val_loader = load_val_data(img_size, batch_size, False)

    # Set logger
    logger = getLogger(__name__)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    os.makedirs(f'./FGSD/{args.model}', exist_ok=True)

    file_handler = logging.FileHandler(f'./FGSD/{args.model}/log.txt', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel('INFO')

    # Cross Entropy loss for FGSD
    criterion = torch.nn.CrossEntropyLoss()

    # encapsule the model
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=None,  # 不更新权重，设置为None
        input_shape=(3, 364, 364),
        nb_classes=200
    )

    base_epsilon = 1 / 255.0 / 0.225
    attack_base = FastGradientMethod(estimator=classifier, eps=base_epsilon)
    all_perturbations = []

    for batch_id, (data, target) in enumerate(tqdm(val_loader, desc="Generating perturbations for the entire validation set")):

        original_data = data.numpy()

        # perform FGSM
        perturbed_data = attack_base.generate(x=original_data)
        perturbations = perturbed_data - original_data

        # Append pertubation
        all_perturbations.append(torch.tensor(perturbations, dtype=torch.float))

    # concat all perturbation tensors
    all_perturbations_tensor = torch.cat(all_perturbations, dim=0)

    # Save all pertubation tensors
    torch.save(all_perturbations_tensor, f'./FGSD/{args.model}/all_perturbations.pt')
    del all_perturbations_tensor, all_perturbations

    # load perturbation tensors
    base_perturbations = torch.load(f'./FGSD/{args.model}/all_perturbations.pt')
    # set epsilon, which is the magnitude of the perturbation
    epsilon_values = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    top_1_accs = []
    top_5_accs = []
    success_attack_rates = []
    number_of_image = 10000  # There are total 10000 images in the validation set

    for epsilon in epsilon_values:
        perturbed_images = []
        true_labels = []
        success_attack_count = 0
        scaled_perturbations = (base_perturbations * epsilon).float()

        for batch_id, (data, target) in enumerate(
                tqdm(val_loader, desc=f'Perform FGSD Attack with Epsilon={epsilon} Progress')):

            start_idx = batch_id * data.size(0)
            end_idx = start_idx + data.size(0)
            perturbed_data = data + scaled_perturbations[start_idx:end_idx].to(data.device)

            # Generate Adversarial Samples
            perturbed_images.append(perturbed_data)
            true_labels.append(target.numpy())

            # Save Adversarial Samples and their perturbations
            for idx, p_image in enumerate(perturbed_data):
                p_image_tensor = p_image.detach()
                perturbation = p_image_tensor - data[idx]

                perturb_dir = f'./FGSD/{args.model}/epsilon_{epsilon}/perturbation/{int(target[idx])}'
                perturbed_dir = f'./FGSD/{args.model}/epsilon_{epsilon}/perturbed_image/{int(target[idx])}'
                os.makedirs(perturb_dir, exist_ok=True)
                os.makedirs(perturbed_dir, exist_ok=True)

                save_image(perturbation, os.path.join(perturb_dir, f'batch_{batch_id}_image_{idx}.png'))
                save_image(p_image_tensor, os.path.join(perturbed_dir, f'batch_{batch_id}_image_{idx}.png'))

                attack_success = visualize(args.model, model, device, data[idx], p_image_tensor, target[idx], batch_id,
                                           idx, epsilon)
                if attack_success:
                    success_attack_count += 1

        # Create Dataloader for Adversarial Samples
        perturbed_images_tensor = torch.tensor(np.vstack(perturbed_images), dtype=torch.float32)
        true_labels_tensor = torch.tensor(np.concatenate(true_labels), dtype=torch.long)
        del true_labels, perturbed_images

        perturbed_dataset = TensorDataset(perturbed_images_tensor, true_labels_tensor)
        perturbed_loader = DataLoader(perturbed_dataset, batch_size=val_loader.batch_size)

        # Save all the p-images and labels for future use.
        pick_path = f'./FGSD/{args.model}/epsilon_{epsilon}/perturbed_val_dataset.pkl'
        pickle_data(perturbed_images_tensor, true_labels_tensor, pick_path)

        # evaluate the model on Adversarial Samples
        val_loss, top_1_acc, top_5_acc = validate(model, device, perturbed_loader, epsilon, can_visualize=False)
        success_attack_rate = success_attack_count / number_of_image
        success_attack_rates.append(success_attack_rate)
        top_1_accs.append(top_1_acc)
        top_5_accs.append(top_5_acc)
        del perturbed_dataset, perturbed_loader
        # log out the info
        logger.info(
            f"Epsilon: {epsilon}, Top-1 Accuracy: {top_1_acc}, Top-5 Accuracy: {top_5_acc}, Attack Success rate: {success_attack_rate}")

    # save important data
    with open(f'./FGSD/{args.model}/metrics.txt', 'w') as file:
        file.write('Epsilon values: ')
        file.write(', '.join(f"{eps:.4f}" for eps in epsilon_values) + '\n')

        file.write('Top-1 Accuracies: ')
        file.write(', '.join(f"{acc:.4f}" for acc in top_1_accs) + '\n')

        file.write('Top-5 Accuracies: ')
        file.write(', '.join(f"{acc:.4f}" for acc in top_5_accs) + '\n')

        file.write('Attack Success Rates: ')
        file.write(', '.join(f"{rate:.4f}" for rate in success_attack_rates) + '\n')

    logger.info("Metrics saved to 'metrics.txt'.")

    # remove handler to prevent redundant log
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)

    # Prepare for the plot
    plt.figure(figsize=(18, 6))

    # Top-1 Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epsilon_values, [top_1_accs[0]] * len(epsilon_values), 'b-', label='Clean Images')
    plt.plot(epsilon_values, top_1_accs, 'g-', label='FGSD Attack')
    plt.xlabel('Epsilon')
    plt.ylabel('Top-1 Accuracy')
    plt.title('Top-1 Accuracy vs Epsilon')
    plt.legend()

    # Top-5 Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epsilon_values, [top_5_accs[0]] * len(epsilon_values), 'b-', label='Clean Images')
    plt.plot(epsilon_values, top_5_accs, 'g-', label='FGSD Attack')
    plt.xlabel('Epsilon')
    plt.ylabel('Top-5 Accuracy')
    plt.title('Top-5 Accuracy vs Epsilon')
    plt.legend()

    # Attack Success Rate
    plt.subplot(1, 3, 3)
    plt.plot(epsilon_values, success_attack_rates, 'r-', label='Attack Success Rate')
    plt.xlabel('Epsilon')
    plt.ylabel('Success Rate')
    plt.title('Attack Success Rate vs Epsilon')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./FGSD/{args.model}/performance_vs_epsilon.png')
