import torch
import torch.nn as nn
from typing import List


class LConditionEncoder(nn.Module):
    """
    Encode L_lat (B,1,h,w) using UNet's down path weights (shared, frozen).
    """
    def __init__(self, unet, in_ch: int = 1):
        super().__init__()
        self.unet = unet
        self.in_proj = nn.Conv2d(in_ch, unet.config.in_channels, kernel_size=3, padding=1)
        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks

    # def forward(self, L_lat: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
    #     x = self.in_proj(L_lat)
    #     x = self.conv_in(x)

    #     feats: List[torch.Tensor] = []
    #     for down in self.down_blocks:
    #         x, _ = down(hidden_states=x, temb=None, encoder_hidden_states=encoder_hidden_states)
    #         feats.append(x)
    #     return feats
    def forward(self, L_lat: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        x = self.in_proj(L_lat)
        x = self.conv_in(x)

        # make timestep batch-shaped
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device)
        timestep = timestep.to(x.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.shape[0] == 1 and x.shape[0] > 1:
            timestep = timestep.expand(x.shape[0])

        t_emb = self.unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.unet.time_embedding(t_emb)

        feats = []
        for down in self.down_blocks:
            x, _ = down(hidden_states=x, temb=emb, encoder_hidden_states=encoder_hidden_states)
            feats.append(x)
        return feats



class MultiLevelCondInjector(nn.Module):
    """
    Trainable 1x1 projections: cond feats -> residuals for each down-block output.
    """
    def __init__(self, unet):
        super().__init__()
        chs = list(unet.config.block_out_channels)  # e.g., [320,640,1280,1280]
        self.proj = nn.ModuleList([nn.Conv2d(chs[i], chs[i], 1) for i in range(len(chs))])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.proj[i](feats[i]) for i in range(len(feats))]


class LConditionedUNetWrapper(nn.Module):
    """
    UNet forward re-implemented to inject residuals after each down block output.
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.lenc = LConditionEncoder(unet)
        self.injector = MultiLevelCondInjector(unet)

        # freeze shared down-path weights inside lenc
        for p in self.lenc.conv_in.parameters():
            p.requires_grad = False
        for p in self.lenc.down_blocks.parameters():
            p.requires_grad = False
        # train ONLY the L->UNet projection
        for p in self.lenc.in_proj.parameters():
            p.requires_grad = True

    def forward(self, sample, timestep, encoder_hidden_states, L_latent_1ch):
        # conditioning features from L
        feats = self.lenc(L_latent_1ch, timestep, encoder_hidden_states)
        cond_res = self.injector(feats)  # trainable

        unet = self.unet

        # time embedding
        t_emb = unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = unet.time_embedding(t_emb)

        # conv in
        h = unet.conv_in(sample)

        # down
        down_block_res_samples = (h,)
        for i, down in enumerate(unet.down_blocks):
            h, res_samples = down(hidden_states=h, temb=emb, encoder_hidden_states=encoder_hidden_states)
            h = h + cond_res[i]
            down_block_res_samples += res_samples

        # mid
        h = unet.mid_block(h, emb, encoder_hidden_states=encoder_hidden_states)

        # up
        for up in unet.up_blocks:
            res_samples = down_block_res_samples[-len(up.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up.resnets)]
            h = up(hidden_states=h, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states)

        # out
        h = unet.conv_norm_out(h)
        h = unet.conv_act(h)
        h = unet.conv_out(h)
        return h
