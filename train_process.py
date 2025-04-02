import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import json
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchinfo import summary
from tqdm import tqdm

from datasets.dataloader import get_dataloader
from models.diffusion import GaussianDiffusion
from models.Unet import UNetModel


def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        model_config = json.load(file)
    return model_config


if __name__=="__main__":
    # configuring
    config_file_path = "configs/config.json"
    model_config = load_config(config_file_path)
    batch_size = model_config["data_settings"]["batch_size"]
    time_steps = model_config["model_settings"]["time_steps"]
    image_size = model_config["data_settings"]["resized_image_size"]
    lr = model_config["training_settings"]["lr"]
    epochs = model_config["training_settings"]["epochs"]
    normal_mean = model_config["data_settings"]["normalization_mean"]
    normal_std = model_config["data_settings"]["normalization_std"]
    channels = model_config["data_settings"]["channels"]
    channel_mul = model_config["model_settings"]["channel_mul_layer"]
    attention_mul = model_config["model_settings"]["attention_mul"]
    num_head = model_config["model_settings"]["num_head"]

    datasets_root = model_config["data_settings"]["root"]
    data_name = model_config["data_settings"]["name"]
    log_root = model_config["path_settings"]["log_dir"]
    log_dir = os.path.join(log_root, data_name)
    model_save_dir = model_config["path_settings"]["weight_save_dir"]
    model_output_path = os.path.join(model_save_dir, "weights_"+data_name+".pth")
    print(f"Configuration Done!")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # dataloader
    transform2Tensor = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                           transforms.Normalize(mean=[normal_mean]*channels, std=[normal_std]*channels)])
    train_dataloader = get_dataloader(data_name, datasets_root, transform2Tensor, batch_size, split="train", download=True)
    print(f"Dataloading Done!")

    unet = UNetModel(input_channels=channels, output_channels=channels, attention_mul=attention_mul, channel_mul_layer=channel_mul, num_head=num_head).to(device=device)
    summary(unet, input_size=[(1, channels, image_size, image_size), (1, 1)])
    diffusion = GaussianDiffusion(device=device, time_steps=time_steps)

    # initialize weights if possible
    if os.path.exists(model_output_path):
        print("Initializing weights of model from existing pth!")
        unet.load_state_dict(torch.load(model_output_path))
    # define optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    # define summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # training
    cur_min_loss = float('inf')
    print(f"---starting training---")
    unet.train()
    for epoch in range(epochs):
        total_loss = 0
        for (images, labels) in tqdm(train_dataloader, desc="training preocess", total=len(train_dataloader)):
            optimizer.zero_grad()
            images = images.to(device)
            t = torch.randint(1, time_steps, (batch_size, 1), device=device)
            loss = diffusion.get_train_loss(unet, images, t)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"{epoch+1}/{epochs}, train loss: {total_loss/len(train_dataloader)}")
        writer.add_scalar("training loss", total_loss/len(train_dataloader), epoch)
        if total_loss<cur_min_loss:
            cur_min_loss = total_loss
            torch.save(unet.state_dict(), model_output_path)
    writer.close()
