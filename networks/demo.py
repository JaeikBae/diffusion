# %%
import torch
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer


# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE)

# %%
tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# %%
# prompt = "A painting of a beautiful sunset over a calm lake"
uncond_prompt = ""
do_cfg = True
cfg_scale = 7 # 1~40

# %%
input_image =  None

# %%
strength = 0.8
sampler = "ddpm"
num_inference_steps = 100
seed = 42

# %%
prompt = "Make car have a red color and a sporty look."
image_path = "../images/00017.jpg"
input_image = Image.open(image_path)

# %%
print("Generating image on device:", DEVICE)
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    torkenizer=tokenizer
)

# %%
Image.fromarray(output_image)

# %%
