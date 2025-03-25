import os
import sys
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))


def time_embedding(time_steps, embedding_dim, frequency=10000):
    """
    create time embedding for each batch
    time_steps: time steps for each batch, has a size (N,)
    embedding_dim: dimension of time embedding(even number required)
    frequency: control the frequency of sin and cos
    return: time embedding with size (N, embedding_dim)
    """
    assert embedding_dim%2==0, "embedding dim is required for even number!"
    time_steps = time_steps.reshape(-1, 1)
    P = torch.zeros((time_steps.shape[0], embedding_dim))
    X = time_steps/torch.pow(frequency, torch.arange(0, embedding_dim, 2, dtype=torch.float32)/embedding_dim)
    P[:, 0::2] = torch.sin(X)
    P[:, 1::2] = torch.cos(X)
    return P


if __name__=="__main__":
    time_steps = torch.arange(1, 10).reshape(-1, 1)
    time = time_embedding(time_steps, 4)
    print(time)


