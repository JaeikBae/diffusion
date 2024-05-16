import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    torkenizer=None,
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1]")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = torkenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids
            # (batch_size, seq_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_length) -> (batch_size, seq_length, dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = torkenizer.batch_encode_plus(
                [uncond_prompt],
                padding="max_length",
                max_length=77,
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_length) -> (batch_size, seq_length, dim)
            uncond_context = clip(uncond_tokens) 

            # (2, Seq_length, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convet it into a lise of tokens
            # Convert the prompt into tokens using the tokenizer
            tokens = torkenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, 3)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, 3) -> (Batch_size, 3, Height, Width)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_size, 3, Height, Width) -> (Batch_size, Height, Width, 3)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device) 

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                # (Batch_size, 4, Latent_Height, Latent_Width) -> (2*Batch_size, 4, Latent_Height, Latent_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond 

            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_size, 3, Height, Width) -> (Batch_size, Height, Width, 3)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, src, dst, clamp=False):
    x = x.float()
    min_src, max_src = src
    min_dst, max_dst = dst
    x -= min_src
    x *= (max_dst - min_dst) / (max_src - min_src)
    x += min_dst
    if clamp:
        x.clamp(min_dst, max_dst)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)