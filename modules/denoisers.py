import torch
from typing import Optional
from diffusers import DDPMScheduler, DDIMScheduler


class Denoiser:
    def __init__(self, scheduler, method: str):
        self.scheduler = scheduler
        self.method = method  # "ddpm" or "ddim"

    @torch.no_grad()
    def sample(
        self,
        model,
        latents: torch.Tensor,                 # (B,4,h,w)
        encoder_hidden_states: torch.Tensor,   # (B,77,768) empty embeddings
        L_latent_1ch: torch.Tensor,            # (B,1,h,w)
        num_steps: int,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Runs denoising loop and returns final latents.
        - DDPM uses scheduler.step(model_output, t, sample)
        - DDIM uses scheduler.step(..., eta=eta)
        """
        device = latents.device
        self.scheduler.set_timesteps(int(num_steps), device=device)

        x = latents
        for t in self.scheduler.timesteps:
            eps = model(
                sample=x,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                L_latent_1ch=L_latent_1ch,
            )

            if self.method == "ddim":
                out = self.scheduler.step(
                    model_output=eps,
                    timestep=t,
                    sample=x,
                    eta=float(eta),
                    generator=generator,
                )
            else:
                out = self.scheduler.step(
                    model_output=eps,
                    timestep=t,
                    sample=x,
                    generator=generator,
                )

            x = out.prev_sample

        return x


def build_denoiser(model_id: str, method: str) -> Denoiser:
    """
    Build denoiser from SD scheduler config:
      - ddpm: DDPMScheduler
      - ddim: DDIMScheduler (from same config)
    """
    base = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    method = method.lower().strip()

    if method == "ddim":
        sched = DDIMScheduler.from_config(base.config)
        return Denoiser(sched, method="ddim")
    elif method == "ddpm":
        sched = DDPMScheduler.from_config(base.config)
        return Denoiser(sched, method="ddpm")
    else:
        raise ValueError(f"Unknown sampling method: {method}")
