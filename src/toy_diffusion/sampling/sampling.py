import torch
from tqdm import tqdm


@torch.no_grad()
def sample_ddim(model, shape, inference_steps=50, temperature=0.0):

    model.eval()

    timesteps = (
        torch.linspace(
            model.timesteps - 1,
            0,
            inference_steps + 1,
            dtype=torch.long,
            device=model.device,
        )
        .round()
        .long()
    )

    noise_schedule = model.get_noise_schedule()

    img = torch.randn(shape, device=model.device)

    for i in tqdm(range(inference_steps), desc="Sampling DDIM"):
        t = timesteps[i].view(-1, 1)
        next_t = timesteps[i + 1].view(-1, 1)

        noise_pred = model(img, t)

        a_bar_t = noise_schedule[t]
        a_bar_next = noise_schedule[next_t]

        pred_x0 = (img - (1 - a_bar_t).sqrt() * noise_pred) / a_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)

        dir_xt = (
            1 - a_bar_next - temperature**2 * (1 - a_bar_next)
        ).sqrt() * noise_pred

        img = (
            torch.sqrt(a_bar_next) * pred_x0
            + dir_xt
            + temperature * torch.sqrt(1 - a_bar_next) * torch.randn_like(img)
        )

    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)

    return img
