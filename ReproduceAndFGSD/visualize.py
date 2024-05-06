import os

from fileio import get_label_mapping
from torchvision.transforms import Normalize

import random, torch
import matplotlib.pyplot as plt


def cut_long_label(str):
    str = str.split(",")
    if len(str) >= 3:
        str = f"{str[0]},{str[1]},{str[2]},..."
    else:
        str = ",".join(str)
    return str


@torch.no_grad()
def visualize(x, pred, y, epoch):
    batch_size = len(y)
    idx = random.randint(0, batch_size - 1)

    prediction = torch.max(pred.data, 1)[1][idx].item()
    actual = y[idx].item()
    inv_normalize = Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
    )
    random_sample = torch.round(torch.mul(inv_normalize(x[idx]), 225)).type(torch.ByteTensor).cpu().permute(1, 2, 0)

    mapping = get_label_mapping()
    plt.imshow(random_sample)
    label_str = cut_long_label(mapping.iloc[actual]['label'])
    prediction_str = cut_long_label(mapping.iloc[prediction]['label'])

    plt.title(f"Label: {label_str}\nPrediction: {prediction_str}", wrap=True, fontsize=12)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f'predictions/val{epoch + 1}_{idx}.png')


@torch.no_grad()
def visualize(model_name, model, device, original_image, p_image, true_label, batch, id_in_batch, epsilon):
    model.eval()
    original_image = original_image.unsqueeze(0).to(device)
    p_image = p_image.unsqueeze(0).to(device)

    # get the perturbed image
    # Predict the classes for the original and perturbed images
    logits_of_original_image = model(original_image)
    logits_of_p_image = model(p_image)
    logits_of_perturbation = model(p_image - original_image)
    # Inverse normalization
    inv_normalize = Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
    )

    # Inverse the images
    original_image = inv_normalize(original_image[0]).cpu().detach()
    p_image = inv_normalize(p_image[0]).cpu().detach()
    perturbation = p_image - original_image
    perturbation = torch.where(perturbation < 0, perturbation + 1, perturbation)

    # get prediction label and probability
    probs_original, preds_original = torch.max(torch.nn.functional.softmax(logits_of_original_image, dim=1), 1)
    probs_perturbed, preds_perturbed = torch.max(torch.nn.functional.softmax(logits_of_p_image, dim=1), 1)
    probs_p, preds_p = torch.max(torch.nn.functional.softmax(logits_of_perturbation, dim=1), 1)

    # Convert label index to name
    mapping = get_label_mapping()
    true_label_str = cut_long_label(mapping.iloc[int(true_label)]['label'])
    pred_label_original_str = cut_long_label(mapping.iloc[preds_original.item()]['label'])
    pred_label_perturbed_str = cut_long_label(mapping.iloc[preds_perturbed.item()]['label'])
    pred_label_p_str = cut_long_label(mapping.iloc[preds_p.item()]['label'])

    # Set up the plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'True Label: {true_label_str}', fontsize=16)

    axs[0].imshow(original_image.permute(1, 2, 0))
    axs[0].set_title(f'Original\n{pred_label_original_str}\nProb: {probs_original.item():.2%}')
    axs[0].axis('off')

    axs[1].imshow(perturbation.permute(1, 2, 0))
    axs[1].set_title(f'Perturbation with Epsilon = {epsilon}\n{pred_label_p_str}\nProb: {probs_p.item():.2%}')
    axs[1].axis('off')

    axs[2].imshow(p_image.permute(1, 2, 0))
    axs[2].set_title(f'Perturbed\n{pred_label_perturbed_str}\nProb: {probs_perturbed.item():.2%}')
    axs[2].axis('off')

    # Save the image
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the global title

    directory = f'FGSD/{model_name}/epsilon_{epsilon}/visualize/{int(true_label)}'
    filename = f'val{batch}_{id_in_batch}.png'

    path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    plt.savefig(path)
    plt.close(fig)

    return logits_of_p_image[:, true_label] < logits_of_original_image[:, true_label]

@torch.no_grad()
def check_attack_success(model, device, original_image, p_image, true_label):
    model.eval()
    original_image = original_image.unsqueeze(0).to(device)
    p_image = p_image.unsqueeze(0).to(device)

    # get the perturbed image
    # Predict the classes for the original and perturbed images
    logits_of_original_image = model(original_image)
    logits_of_p_image = model(p_image)

    # # get prediction label and probability
    top1_pred = torch.max(logits_of_p_image, 1)[1]
    top1_correct = (top1_pred == true_label)

    # check the index of the top5 possible labels
    top5_preds = torch.topk(logits_of_p_image, 5, dim=1)[1]
    top5_correct = true_label in top5_preds[0]  # Assuming batch size of 1

    return top1_correct, top5_correct, logits_of_p_image[:, true_label] < logits_of_original_image[:, true_label]
