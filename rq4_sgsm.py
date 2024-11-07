#This script has code snippets taken from https://github.com/rolux/flux-random-walk/blob/main/flux-random-walk.ipynb
import numpy as np
from PIL import Image
import torch
from scipy.stats import chi
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import json
import argparse
import os
import sys
from torchvision import transforms
from driving_resnet import ResnetModel

sys.path.append('./ComfyUI')

from nodes import NODE_CLASS_MAPPINGS
from comfy import model_management
from comfy_extras import nodes_custom_sampler, nodes_flux

model = "dev" # "schnell" or "dev"
load_encoder = True
OUTPUT_DIR = "./ComfyUI/content/outputs"

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1000, help='no of samples')
parser.add_argument('--req', type=str, default="r1", help='requirement')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--max_iterations', type=int, default=10, help='max number of iterations')
args = parser.parse_args()

with open('./loras_prompts.json') as json_file:
    reqs = json.load(json_file)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

with torch.inference_mode():
    if load_encoder:
        clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.sft", "clip_l.sft", "flux")[0]
    unet = UNETLoader.load_unet(f"flux1-{model}-fp8.sft", "fp8_e4m3fn")[0]
    vae = VAELoader.load_vae("ae.sft")[0]
    unet, clip = LoraLoader.load_lora(unet, clip, reqs[args.req]["lora_name"], 1.0, 1.0)


def encode(prompt):
    with torch.inference_mode():
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond, pooled

def render(
    filename,
    prompt, # tuple of tensors ((1, 256, 4096), (1, 768))
    noise,  # tensor (16, height//8, width//8)
    steps=4 if model == "schnell" else 20,
    guidance=3.5
):
    if os.path.exists(filename):
        return
    print(filename.replace(f"{OUTPUT_DIR}/", ""))
    cond = [[prompt[0], {"pooled_output": prompt[1]}]]
    width, height = noise.shape[2] * 8, noise.shape[1] * 8

    with torch.inference_mode():
        cond = FluxGuidance.append(cond, guidance)[0]
        random_noise = RandomNoise.get_noise(noise)[0]
        guider = BasicGuider.get_guider(unet, cond)[0]
        sampler = KSamplerSelect.get_sampler("euler")[0]
        sigmas = BasicScheduler.get_sigmas(unet, "simple", steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(width, height)[0]
        sample, sample_denoised = SamplerCustomAdvanced.sample(
            random_noise, guider, sampler, sigmas, latent_image
        )
        model_management.soft_empty_cache()
        decoded = VAEDecode.decode(vae, sample)[0].detach()
        image = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)
    return image

width, height = 900, 256
steps, sigma = args.size, 3
plot_noise = True
seed = args.seed
max_iterations = args.max_iterations
p_temp =  reqs[args.req]["prompt"]           
print(f"req: {args.req} prompt: {p_temp} seed: {seed} size: {args.size}")

dirname = f"{args.req}_{seed}"
shutil.rmtree(f"{OUTPUT_DIR}/random_walk/{dirname}", ignore_errors=True)

cond_pooled = (encode(reqs[args.req]["prompt"]) for _ in tqdm(range(steps)))
cond, pooled = zip(*cond_pooled)
    

model = ResnetModel()
# Load weights
model.load_state_dict(torch.load("./output/driving_model.ckpt", map_location=torch.device('cpu')))
model.eval()
model.to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

pred = []
with open(f'results/sgsm_{args.req}_passrate.txt', 'w') as ac:
    ac.write(f"Pass Rate for {args.req}\n")
for mi in range(max_iterations):
    g = torch.Generator().manual_seed(seed)
    init = torch.randn((steps, 16, height//8, width//8), generator=g)
  
    for i in range(steps):
        prompt = cond[i], pooled[i]
        noise = init[i]
        render(
            f"{OUTPUT_DIR}/random_walk/{dirname}/{(i+steps*mi):08d}.png",
            prompt,
            noise
        )
    with torch.no_grad():
        for i in range(steps):
            img = Image.open(f"{OUTPUT_DIR}/random_walk/{dirname}/{(i+steps*mi):08d}.png")
            img_tensor = transform(img).to(device)
            batch = img_tensor.unsqueeze(0)
            out = model(batch)

            if args.req in ["r1", "r2"]:
                p = (out['acceleration_continuous'] <= -0.25)
            elif args.req == "r3":
                p = (out['acceleration_continuous'] > -0.25)
            elif args.req in ["r4", "r7"]:
                p = (out['steering_angle'] > -0.07)
            else:
                p = (out['steering_angle'] < 0.07)
                
            pred.append(p.int().item())
            
    print(f"Iteration {mi} Accuracy for {args.req} is {sum(pred)/len(pred)} ")

    with open(f'results/sgsm_{args.req}_passrate.txt', 'a') as ac:
        ac.write(f"Iteration {mi} Pass Rate for {args.req} is {sum(pred)/len(pred)}\n")
    
    seed += 1
