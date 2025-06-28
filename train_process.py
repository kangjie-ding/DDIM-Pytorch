import os
import torch
import torch.amp
import torch.nn as nn
import torch.utils
import torch.utils.data
import json
from contextlib import nullcontext
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

    # model settings
    time_steps = model_config["model_settings"]["time_steps"]
    channel_mul = model_config["model_settings"]["channel_mul_layer"]
    attention_mul = model_config["model_settings"]["attention_mul"]
    num_head = model_config["model_settings"]["num_head"]
    add_2d_rope = model_config["model_settings"]["add_2d_rope"]

    # training settings
    lr = model_config["training_settings"]["lr"]
    epochs = model_config["training_settings"]["epochs"]
    amp_dtype = model_config["training_settings"]["amp_dtype"]
    accumulation_steps = model_config["training_settings"]["accumulation_steps"]
    grad_clip_norm = model_config["training_settings"]["grad_clip_norm"]
    

    # data settings
    image_size = model_config["data_settings"]["resized_image_size"]
    batch_size = model_config["data_settings"]["batch_size"]
    normal_mean = model_config["data_settings"]["normalization_mean"]
    normal_std = model_config["data_settings"]["normalization_std"]
    channels = model_config["data_settings"]["channels"]
    datasets_root = model_config["data_settings"]["root"]
    data_name = model_config["data_settings"]["name"]

    # paths settings
    log_root = model_config["path_settings"]["log_dir"]
    log_dir = os.path.join(log_root, data_name)
    model_save_dir = model_config["path_settings"]["weight_save_dir"]
    model_output_path = os.path.join(model_save_dir, f"weights_{data_name}_{time_steps}.pth")

    print(f"Configuration Done!")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast()
    # fix random seed
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # dataloader
    transform2Tensor = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                           transforms.Normalize(mean=normal_mean, std=normal_std)])
    train_dataloader = get_dataloader(data_name, datasets_root, transform2Tensor, batch_size, split="train", download=True)
    print(f"Dataloading Done!")
    # loading models
    unet = UNetModel(input_channels=channels, output_channels=channels, attention_mul=attention_mul, channel_mul_layer=channel_mul,
                      num_head=num_head, add_2d_rope=add_2d_rope).to(device=device)
    summary(unet, input_size=[(1, channels, image_size, image_size), (1, 1)])
    diffusion = GaussianDiffusion(device=device, time_steps=time_steps)

    # initialize weights if possible
    if os.path.exists(model_output_path):
        print("Initializing weights of model from existing pth!")
        unet.load_state_dict(torch.load(model_output_path))

    # define optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype in ["float16", "bfloat16"])

    # define summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # training
    cur_min_loss = float('inf')
    print(f"---starting training---")
    unet.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (images, labels) in tqdm(enumerate(train_dataloader), desc="training preocess", total=len(train_dataloader), dynamic_ncols=True):
            images = images.to(device)
            t = torch.randint(1, time_steps, (batch_size, 1), device=device)
            with ctx:
                loss = diffusion.get_train_loss(unet, images, t)
                total_loss += loss.item()
                loss /= accumulation_steps

            scaler.scale(loss).backward()

            if (step+1)%accumulation_steps==0 or step+1==len(train_dataloader):
                scaler.unscale_(optimizer) # unscaling gradient
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip_norm) # gradient clipping
                scaler.step(optimizer) # optimizer.step()
                scaler.update()  # adjust scaler dynamically
                optimizer.zero_grad(set_to_none=True)

        print(f"{epoch+1}/{epochs}, train loss: {total_loss/len(train_dataloader)}")
        writer.add_scalar("training loss", total_loss/len(train_dataloader), epoch)
        if total_loss<cur_min_loss:
            cur_min_loss = total_loss
            torch.save(unet.state_dict(), model_output_path)
    writer.close()
