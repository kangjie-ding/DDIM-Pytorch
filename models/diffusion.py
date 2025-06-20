import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

from models.Unet import UNetModel
from utils.schedule_calculation import *

class GaussianDiffusion():
    def __init__(self, device, time_steps=1000, beta_method='linear'):
        super().__init__()
        self.time_steps = time_steps
        self.beta_schedule = get_beta_schedule(time_steps, beta_method).to(device=device)
        self.alpha_schedule = get_alpha_schedule(self.beta_schedule)
        self.cumprod_alpha_schedule = get_cumprod_alpha(self.alpha_schedule)
        self.sqrt_cumprod_alpha_schedule = get_sqrt_cumprod_alpha(self.cumprod_alpha_schedule)
        self.sqrt_1_minus_cumprod_alpha_schedule = get_sqrt_1_minus_cumprod_alpha(self.cumprod_alpha_schedule)

    def q_xt_x0(self, x0, t, noise):
        assert x0.shape==noise.shape, "Different shape for x0 and noise!"
        # since index of shcedule started from 0
        t = t-1
        sqrt_cumprod_alpha = self.sqrt_cumprod_alpha_schedule[t].reshape(-1, 1, 1, 1)
        sqrt_1_minus_cumprod_alpha = self.sqrt_1_minus_cumprod_alpha_schedule[t].reshape(-1, 1, 1, 1)
        # print(sqrt_cumprod_alpha.shape)
        # print(sqrt_1_minus_cumprod_alpha.shape)
        return sqrt_cumprod_alpha*x0 + sqrt_1_minus_cumprod_alpha*noise
    
    def _pred_x0_from_noise(self, xt, t, pred_noise):
        t = t-1
        a = (1./self.sqrt_cumprod_alpha_schedule[t]).reshape(-1, 1, 1, 1)
        b = (self.sqrt_1_minus_cumprod_alpha_schedule[t]/self.sqrt_cumprod_alpha_schedule[t]).reshape(-1, 1, 1, 1)
        return a*xt-b*pred_noise

    def _pred_pre_xt(self, xt, pred_x0, t):
        assert (t>1).all(), "time step is required to >1 in this denoising fashion"
        t = t-1
        pre_t = t-1
        a = (self.sqrt_cumprod_alpha_schedule[pre_t]*self.beta_schedule[t]/(1-self.cumprod_alpha_schedule[t])).reshape(-1, 1, 1, 1)
        b = (torch.sqrt(self.alpha_schedule[t])*(1-self.cumprod_alpha_schedule[pre_t])/(1-self.cumprod_alpha_schedule[t])).reshape(-1, 1, 1, 1)
        mean = a*pred_x0 + b*xt
        variance = ((1-self.cumprod_alpha_schedule[pre_t])*self.beta_schedule[t]/(1-self.cumprod_alpha_schedule[t])).reshape(-1, 1, 1, 1)
        noise = torch.randn(xt.shape).to(device=xt.device)
        return mean + torch.sqrt(variance)*noise

    @torch.no_grad()
    def ddpm_fashion_sample(self, model, image_size, channels, batch_size=16):
        device = next(model.parameters()).device
        xt = torch.randn((batch_size, channels, image_size, image_size)).to(device=device)
        for i in tqdm(range(1, self.time_steps+1)[::-1], desc="denoising process in DDPM fashion", total=self.time_steps):
            t = torch.full((batch_size, 1), i).to(device=device)
            pred_noise = model(xt, t)
            if i!=1:
                xt = self._pred_pre_xt(xt, self._pred_x0_from_noise(xt, t, pred_noise), t)
            else:
                xt = self._pred_x0_from_noise(xt, t, pred_noise)
        return xt

    @torch.no_grad()
    def ddim_fashion_sample(self, model, image_size, channels, sampling_steps=50, sampling_method='uniform', batch_size=16, eta=0.0):
        device = next(model.parameters()).device
        if sampling_method=='uniform':
            assert sampling_steps<=self.time_steps, "Invalid sampling steps in DDIM, must less than time steps training in DDPM!"
            assert self.time_steps%sampling_steps==0, "Invalid sampling steps in DDIM, must be divisible by time steps training in DDPM!"
            step_length = self.time_steps//sampling_steps
            ddim_time_schedule = range(1, self.time_steps+1, step_length)
            ddim_pre_time = [0]+list(ddim_time_schedule)[:-1]
        else:
            raise NotImplementedError("Other sampling methods are going to implement.")
        xt = torch.randn((batch_size, channels, image_size, image_size)).to(device=device)
        for i in tqdm(range(0, sampling_steps)[::-1], desc="denoising process in DDIM fashion", total=sampling_steps):
            t = torch.full((batch_size, 1), ddim_time_schedule[i]).to(device=device)
            pre_t = torch.full((batch_size, 1), ddim_pre_time[i]).to(device=device) if i!=0 else None
            pred_noise = model(xt, t)
            pred_x0 = self._pred_x0_from_noise(xt, t, pred_noise)
            if i!=0:
                sigma_t = eta*torch.sqrt((1-self.cumprod_alpha_schedule[pre_t-1])/(1-self.cumprod_alpha_schedule[t-1]))*torch.sqrt(1-self.cumprod_alpha_schedule[t-1]/self.cumprod_alpha_schedule[pre_t-1])
                sigma_t = sigma_t.reshape(-1, 1, 1, 1)
                dir_point_xt = torch.sqrt(1-self.cumprod_alpha_schedule[pre_t-1].reshape(-1, 1, 1, 1)-sigma_t**2)*pred_noise
                random_noise = torch.rand_like(sigma_t)
                xt = torch.sqrt(self.cumprod_alpha_schedule[pre_t-1]).reshape(-1, 1, 1, 1)*pred_x0+dir_point_xt+sigma_t*random_noise
            else:
                xt = pred_x0
        return xt

    def get_train_loss(self, model, x0, t):
        noise = torch.randn(x0.shape, device=x0.device)
        xt = self.q_xt_x0(x0, t, noise)
        pred_noise = model(xt, t)
        return F.mse_loss(noise, pred_noise)
    
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x0 = torch.randn((16, 3, 64, 64)).to(device=device)
    # t = torch.randint(1, 1000, (16, 1)).to(device=device)
    unet = UNetModel(num_head=1, channel_mul_layer=(1, 2, 2, 2), model_channels=32).to(device=device)

    diffusion = GaussianDiffusion(device=device, time_steps=1000)
    # loss = diffusion.get_train_loss(unet, x0, t)
    # print(loss)
    #img = diffusion.ddpm_fasion_sample(unet, 64, 3, 16)
    img = diffusion.ddim_fashion_sample(unet, 32, 3, 50)
    print(img.shape)



    