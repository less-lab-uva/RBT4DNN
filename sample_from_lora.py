import torch
from diffusers import FluxPipeline
import os
import glob
import argparse
import random
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid


def make_image_panel(im_path, ori_path, num_sample, img_name, save_loc, img_size):
    ims = []

    fnames = glob.glob(os.path.join(ori_path, '*.{}'.format('png')))
    fnames += glob.glob(os.path.join(ori_path, '*.{}'.format('jpg')))
    fnames += glob.glob(os.path.join(ori_path, '*.{}'.format('jpeg')))

    fnames = random.sample(fnames, k=num_sample)
    for fname in fnames:
        ims.append(fname)
    
    fnames = glob.glob(os.path.join(im_path, '*.{}'.format('png')))
    fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpg')))
    fnames += glob.glob(os.path.join(im_path, '*.{}'.format('jpeg')))

    fnames = random.sample(fnames, k=num_sample)
    for fname in fnames:
        ims.append(fname)
        
    print("ims: ",len(ims))

    if not os.path.exists(save_loc):
        os.mkdir(save_loc)


    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(img_size),
                    # torchvision.transforms.CenterCrop(img_size),
                    torchvision.transforms.ToTensor(),
                ])

    img = []
    for index in tqdm(range(len(ims))):
        im = Image.open(ims[index])
        im = transform(im)
        img.append(im)
    print("processed image: ",len(ims))
    grid = make_grid(img, nrow=num_sample)
    img = torchvision.transforms.ToPILImage()(grid)
        
        
    img.save(f'{save_loc}/{img_name}.png')
    img.close()

def mnist_single_lora(req, isreqdataonly = False):
    if isreqdataonly:
        weight_path = f"output/flux_lora_mnist_all_req/flux_lora_mnist_all_req.safetensors"
        im_path = f"output/flux_lora_mnist_all_req"
    else:
        weight_path = f"output/flux_lora_mnist_all_data/flux_lora_mnist_all_data.safetensors"
        im_path = f"output/flux_lora_mnist_all_data"
    #prompts
    prompt = {
        'r1': 'MNIST hand written digit in white on black background. The digit is a 2 and has very low height.',
        'r2': 'MNIST hand written in white on black background. The digit is a 3 and is very thick.',
        'r3': 'MNIST hand written in white on black background. The digit is a 7 and is very thick.',
        'r4': 'MNIST hand written digit in white on black background. The digit is a 9 and is very left leaning.',
        'r5': 'MNIST hand written digit in white on black background. The digit is a 6 and is very right leaning.',
        'r6': 'MNIST hand written in white on black background. The digit is a 0 and has very low height.'
    }


    
    img_size = 64
    return weight_path, prompt[req], im_path, img_size 
def mnist(req):
    weight_path = f"output/flux_lora_mnist_{req}/flux_lora_mnist_{req}.safetensors"
    prompt = {
        'r1': 'MNIST hand written digit in white on black background. The digit is a 2 and has very low height.',
        'r2': 'MNIST hand written digit in white on black background. The digit is a 3 and is very thick.',
        'r3': 'MNIST hand written digit in white on black background. The digit is a 7 and is very thick.',
        'r4': 'MNIST hand written digit in white on black background. The digit is a 9 and is very left leaning.',
        'r5': 'MNIST hand written digit in white on black background. The digit is a 6 and is very right leaning.',
        'r6': 'MNIST hand written digit in white on black background. The digit is a 0 and has very low height.'
    }



    im_path = f"output/flux_lora_mnist_{req}"
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    img_size = 64
    return weight_path, prompt[req], im_path, img_size

def celeba(req):
    weight_path = f"output/flux_lora_celeba_{req}/flux_lora_celeba_{req}.safetensors"
    prompt = {
        'r1': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has black hair.',
        'r2': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has brown hair.',
        'r3': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has a mustache.',
        'r4': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and has wavy hair.',
        'r5': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and is bald.',
        'r6': 'CelebAHQ close headshot of a person. The person is wearing eyeglasses and hat.',
    }
    
    img_size = 1024
    im_path = f"output/flux_lora_celeba_{req}"
    if not os.path.exists(im_path):
        os.mkdir(im_path)
   
    
    return weight_path, prompt[req], im_path, img_size

def sgsm(req):
    weight_path = f"output/flux_lora_sgsm_{req}/flux_lora_sgsm_{req}.safetensors"
    prompt ={
        "r1": "C4RL4. Vehicle within 10 meters, in the same lane and in front of (R1)",
        "r2": "C4RL4. Ego lane controlled by yellow/red light (R2)",
        "r3": "C4RL4. Ego lane is controlled by a green light and no entity is in front and in the same lane and within 10 meters (R3)",
        "r4": "C4RL4. Ego vehicle in the rightmost lane and not in a junction (R4)",
        "r5": "C4RL4. Ego vehicle in the leftmost lane and not in a junction (R5)",
        "r6": "C4RL4. Vehicle in the lane to the left and within 7 meters (R6)",
        "r7": "C4RL4. Vehicle in the lane to the right and within 7 meters (R7)"
    }
    im_path = f"output/flux_lora_sgsm_{req}"
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    img_size = [912, 256]
    return weight_path, prompt[req], im_path, img_size
def sample_images(req, pipe, dataset, num_sample, num_images_per_epoch, isSingleLora = False, isReqDataOnly = False):

    #Load the LoRA adapter with PEFT backend
    if args.isSingleLora:
        if 'mnist' in dataset:
            weight_path, prompt, im_path, img_size = mnist_single_lora(req, isreqdataonly = args.isReqDataOnly)
        elif 'celeba' in dataset:
            weight_path, prompt, im_path, img_size = celeba(req)
        im_dir = req
    else:
        if 'mnist' in dataset:
            weight_path, prompt, im_path, img_size = mnist(req)
            width = img_size
            height = img_size
        elif 'celeba' in dataset:
            weight_path, prompt, im_path, img_size = celeba(req)
            width = img_size
            height = img_size
        elif 'sgsm' in dataset:
            weight_path, prompt, im_path, img_size = sgsm(req)
            width = img_size[0]
            height = img_size[1]
    
    if not os.path.exists(im_path):
        os.mkdir(im_path)



    #Generate image

    num_epoch = int(num_sample/num_images_per_epoch)
    if num_sample%num_images_per_epoch!=0:
        num_epoch = num_epoch + 1
        print(f"total num of samples to generate: ", num_epoch*num_images_per_epoch)
    print(f'num_epoch: ', num_epoch)
    
    with torch.no_grad():
        for i in range(num_epoch):
            print("i: ",i)
            seed = random.randint(0, 9999)
            image = pipe(prompt, height=height, width=width, num_images_per_prompt = num_images_per_epoch, generator=torch.Generator("cpu").manual_seed(seed))
            # print("img: ",len(image.images))
            for j in range(len(image.images)):
                # print("j: ",j)
                img = image.images[j]
                # Crop first and last 6 pixels from the PIL image width for sgsm
                if 'sgsm' in dataset:
                    img = img.crop((6, 0, 906, 256))
                #save the image
                img.save(f"{im_path}/{i*len(image.images)+j}.png")
                img.close()


def main(args):
    req = args.req
    dataset = args.dataset
    #Load the main Flux model
    pipe = FluxPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        torch_dtype = torch.bfloat16
    ).to("cuda")


    

    print("model loading done")
    for r in req:
        print(f'generating image for {r}')
        if args.isSingleLora:
            if 'mnist' in dataset:
                weight_path,_, _, _ = mnist_single_lora(r, isreqdataonly = args.isReqDataOnly)
            elif 'celeba' in dataset:
                weight_path, _, _, _ = celeba(r)
        else:
            if 'mnist' in dataset:
                weight_path,_, _, _ = mnist(r)
            elif 'celeba' in dataset:
                weight_path, _, _, _ = celeba(r)
            elif 'sgsm' in dataset:
                weight_path, _, _, _ = sgsm(r)
        if weight_path is not None:
            pipe.load_lora_weights(
                weight_path,
                adapter_name = f"default_{r}"
            )
            print("weight loading done")
        #Enable model offloading if necessary
        pipe.enable_model_cpu_offload()
        ####for single model
        sample_images(r, pipe, dataset, args.num_samples, args.num_samples_per_epoch, args.im_dir, args.isSingleLora, args.isReqDataOnly)
        # sample_images(r, pipe, dataset, args.num_samples, args.num_samples_per_epoch, args.im_dir)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq1')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--isSingleLora', default=False, type=bool)
    parser.add_argument('--isReqDataOnly', default=False, type=bool)
    parser.add_argument('--num_samples', default = 1000, type = int)
    parser.add_argument('--num_samples_per_epoch', default = 100, type = int)
    parser.add_argument('--req', nargs='+', default = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
    args = parser.parse_args()

    main(args)
