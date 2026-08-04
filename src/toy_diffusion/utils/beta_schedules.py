import torch


def linear_beta_schedule(timestamps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timestamps)


def cosine_beta_schedule(timestamps, s=0.008):
    t = torch.arange(timestamps + 1)
    f_x = torch.cos((t / timestamps + s) / (1 + s) * torch.pi / 2) ** 2
    a_t = f_x / f_x[0]
    B_t = 1 - a_t[1:] / (a_t[:-1])
    return torch.clip(B_t, 1e-4, 0.9999)
