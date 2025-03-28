import os
import sys
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

__all__ = ['get_beta_schedule', 'get_alpha_schedule', 'get_cumprod_alpha', 'get_sqrt_cumprod_alpha', 'get_sqrt_1_minus_cumprod_alpha']

def get_beta_schedule(time_steps=1000, method='linear'):
    assert method in ['linear'], "Unknown method to get beta schedule" # we can add other methods
    if method=='linear':
        return torch.linspace(0.0001, 0.02, time_steps)
    
def get_alpha_schedule(beta_schedule):
    return 1-beta_schedule

def get_cumprod_alpha(alpha_schedule):
    return torch.cumprod(alpha_schedule, dim=0)

def get_sqrt_cumprod_alpha(cumprod_alpha):
    return torch.sqrt(cumprod_alpha)

def get_sqrt_1_minus_cumprod_alpha(cumprod_alpha):
    return torch.sqrt(1-cumprod_alpha)


if __name__=="__main__":
    betas = get_beta_schedule()
    alphas = get_alpha_schedule(betas)
    cumprod_alpha = get_cumprod_alpha(alphas)
    print(cumprod_alpha.shape)