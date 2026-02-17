# from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0


# def enable_lora_on_unet(unet, rank: int = 8):
#     attn_procs = {}
#     for name in unet.attn_processors.keys():
#         cross_attention_dim = getattr(unet.config, "cross_attention_dim", None)

#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#         elif name.startswith("up_blocks"):
#             block_id = int(name.split(".")[1])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#         elif name.startswith("down_blocks"):
#             block_id = int(name.split(".")[1])
#             hidden_size = unet.config.block_out_channels[block_id]
#         else:
#             hidden_size = unet.config.block_out_channels[-1]

#         try:
#             attn_procs[name] = LoRAAttnProcessor2_0(
#                 hidden_size=hidden_size,
#                 cross_attention_dim=cross_attention_dim,
#                 rank=rank,
#             )
#         except Exception:
#             attn_procs[name] = LoRAAttnProcessor(
#                 hidden_size=hidden_size,
#                 cross_attention_dim=cross_attention_dim,
#                 rank=rank,
#             )

#     unet.set_attn_processor(attn_procs)
#     return unet


from peft import LoraConfig

def enable_lora_on_unet(unet, rank: int = 8):
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    unet.add_adapter(cfg)
    return unet

