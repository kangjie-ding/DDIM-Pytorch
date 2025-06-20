import json
import os
import torch
from models.diffusion import GaussianDiffusion
from models.Unet import UNetModel
from torchvision import transforms
import matplotlib.pyplot as plt

def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        model_config = json.load(file)
    return model_config

if __name__=="__main__":
    transform2Image = transforms.ToPILImage()
    # configuring
    config_file_path = "configs/config.json"
    model_config = load_config(config_file_path)
    channels = model_config["data_settings"]["channels"]
    channel_mul = model_config["model_settings"]["channel_mul_layer"]
    num_head = model_config["model_settings"]["num_head"]
    attention_mul = model_config["model_settings"]["attention_mul"]
    time_steps = model_config["model_settings"]["time_steps"]
    normal_mean = model_config["data_settings"]["normalization_mean"]
    normal_std = model_config["data_settings"]["normalization_std"]
    model_save_dir = model_config["path_settings"]["weight_save_dir"]
    data_name = model_config["data_settings"]["name"]
    image_size = model_config["data_settings"]["resized_image_size"]
    model_output_path = os.path.join(model_save_dir, "weights_"+data_name+".pth")
    print("Configuring Done!")

    # model defining
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    diffusion = GaussianDiffusion(device=device, time_steps=time_steps)
    unet = UNetModel(input_channels=channels, output_channels=channels, channel_mul_layer=channel_mul, num_head=num_head, attention_mul=attention_mul).to(device=device)
    # weights loading
    assert os.path.exists(model_output_path), f"{model_output_path} is not existing!"
    unet.load_state_dict(torch.load(model_output_path))
    # denoising process
    unet.eval()
    generated_images = diffusion.ddim_fashion_sample(unet, image_size, channels, batch_size=16, sampling_steps=100)
    # generated_images = diffusion.ddpm_fashion_sample(unet, image_size, channels, batch_size=16)
    recoverd_images = generated_images*torch.tensor(normal_std, device=device).view(1, -1, 1, 1)+torch.tensor(normal_mean, device=device).view(1, -1, 1, 1)
    # images saving
    images_output_dir = "images_output/"
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir, exist_ok=True)
    for index, image in enumerate(recoverd_images):
        image = transform2Image(image)
        save_path = images_output_dir+f"image_{index}.png"
        image.save(save_path)
    # images showing
    recoverd_images = recoverd_images.reshape(4, 4, channels, image_size, image_size)
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)

    for row in range(4):
        for col in range(4):
            f_ax = fig.add_subplot(gs[row, col])
            image = transform2Image(recoverd_images[row][col])
            f_ax.imshow(image, cmap="gray")
            f_ax.axis("off")
    plt.show()
