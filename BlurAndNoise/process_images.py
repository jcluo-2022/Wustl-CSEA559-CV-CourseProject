import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# Gussain Blur function
def blur_image(image_array, blur_radius=5):
    blurred_image = cv2.GaussianBlur(image_array, (blur_radius, blur_radius), 0)
    return blurred_image


# Noise-adding function
def add_gaussian_noise(image_tensor, mean=0., std=0.1):
    noise = torch.randn(image_tensor.size()) * std + mean
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0., 1.)
    return noisy_image


# Process every image
def process_images(input_folder_path, output_folder_blur, output_folder_noise, blur_radius=5, mean=0., std=0.1):

    # check output directory
    os.makedirs(output_folder_blur, exist_ok=True)
    os.makedirs(output_folder_noise, exist_ok=True)

    transform_to_tensor = transforms.ToTensor()
    transform_to_pil = transforms.ToPILImage()

    # Iterate every image
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".JPEG"):
            image_path = os.path.join(input_folder_path, filename)
            image = cv2.imread(image_path)

            # Blur the image
            blurred_image = blur_image(image, blur_radius)
            cv2.imwrite(os.path.join(output_folder_blur, filename), blurred_image)

            # Adding noise
            image_pil = Image.open(image_path)
            image_tensor = transform_to_tensor(image_pil)

            # Save the noised images
            noisy_image_tensor = add_gaussian_noise(image_tensor, mean, std)
            noisy_image_pil = transform_to_pil(noisy_image_tensor)
            noisy_image_pil.save(os.path.join(output_folder_noise, filename))


# setting path
input_folder_path = "./val/images"  # folder path of original images
output_folder_blur = "./val_blur/images"  # folder path of saving blurred image
output_folder_noise = "./val_noise/images"  # folder path of saving noised image

if __name__ == '__main__':
    # 调用函数
    process_images(input_folder_path, output_folder_blur, output_folder_noise)
