import torch
from diffusers import FluxPipeline
import os
import glob
import random
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
import argparse

def mnist(req):
    weight_path = f"output/flux_lora_mnist_{req}/flux_lora_mnist_{req}.safetensors"
    #prompts
    prompt = {
        'r1': 'MNIST hand written digit in white on black background. The digit is a 2 and has very low height.',
        'r2': 'MNIST hand written in white on black background. The digit is a 3 and is very thick.',
        'r3': 'MNIST hand written in white on black background. The digit is a 7 and is very thick.',
        'r4': 'MNIST hand written digit in white on black background. The digit is a 9 and is very left leaning.',
        'r5': 'MNIST hand written digit in white on black background. The digit is a 6 and is very right leaning.',
        'r6': 'MNIST hand written in white on black background. The digit is a 0 and has very low height.'
    }

    im_path = f"rq4/output/flux_lora_mnist_{req}"
    img_size = 64
    return weight_path, prompt[req], im_path, img_size

def celeba(req):
    weight_path = f"output/flux_lora_celeba_{req}/flux_lora_celeba_{req}.safetensors"
    #prompts
    prompt = {
        'r1': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has black hair.',
        'r2': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has brown hair.',
        'r3': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has straight hair.',
        'r4': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has wavy hair.',
        'r5': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and is bald.',
        'r6': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and hat.'
    }
    
    img_size = 512
    im_path = f"rq4/output/flux_lora_celeba_{req}"
    return weight_path, prompt[req], im_path, img_size

def sample_images(req, dataset, num_sample, num_images_per_epoch, im_dir= 'sample_gen'):
    
    ##Load the main Flux model
    pipe = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype = torch.float16
    ).to("cuda")

    print("model loading done")
    
    #Load the LoRA adapter with PEFT backend
    if 'mnist' in dataset:
        weight_path,prompt,im_path, img_size = mnist(req)
    elif 'celeba' in dataset:
        weight_path, prompt, im_path, img_size = celeba(req)

    im_path = os.path.join(im_path,im_dir)
    
    pipe.load_lora_weights(
        weight_path,
        adapter_name = "default"
    )

    print("weight loading done")
    #Enable model offloading if necessary
    pipe.enable_model_cpu_offload()
    
    if not os.path.exists(im_path):
        os.mkdir(im_path)

    #Generate image

    num_epoch = int(num_sample/num_images_per_epoch)
    if num_sample%num_images_per_epoch!=0:
        num_epoch = num_epoch + 1
    print(f"total num of samples to generate: ", num_epoch*num_images_per_epoch)
    print(f'num_epoch: ', num_epoch)
    
    with torch.no_grad():
        for i in range(0, num_epoch):    #######
            print("i: ",i)
            seed = random.randint(0, 9999)  ########
            #seed = random.randint(10000, 20000)
            image = pipe(prompt, height=img_size, width=img_size,num_images_per_prompt = num_images_per_epoch, generator=torch.Generator("cpu").manual_seed(seed))

            for j in range(len(image.images)):
                img = image.images[j]
                #save the image
                img.save(f"{im_path}/{i*len(image.images)+j}.png")
                img.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--req", type=str, choices=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
    parser.add_argument("--dataset", type=str, choices=["mnist", "celeba"], default="mnist")
    args = parser.parse_args()
    
    req = [args.req]
    dataset = args.dataset
    
    for r in req:
        sample_images(r, dataset, 10000, 100,'rq4_samples')
