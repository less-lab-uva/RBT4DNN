import os
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from torchvision import transforms
from scipy.spatial import distance
from scipy.stats import entropy
from torch.utils.data import DataLoader
from rq1.train_classifier import ResnetModel, CustomDataset


def find_index(df, feature1, feature2 = None):
    index = []
    for i in df.index:
        if feature2 is not None:
            if df[feature1].loc[i] == 1 and df[feature2].loc[i] == 1:
                index.append(i)
        else:
            if df[feature1].loc[i] == 1:
                index.append(i)
    return index

def load_model(model_name, device, is_sgsm = False):
    model = ResnetModel(num_of_class = 2, is_sgsm = is_sgsm).to(device)
    model.load_state_dict(torch.load(model_name, weights_only = True))
    model.eval()
    return model
def calculate_bin_using_GTC(im_path, model, transform, device, im_channel = 3, im_index = None, is_sgsm = False):
    if im_index is not None:
        dataset = CustomDataset(im_names = im_index, label = None, transform = transform, 
                        im_path = im_path, feature = None, im_channel = im_channel)
        index = im_index
    else:
        index = glob.glob(os.path.join(im_path, '*.{}'.format('png')))
        index += glob.glob(os.path.join(im_path, '*.{}'.format('jpg')))
        index += glob.glob(os.path.join(im_path, '*.{}'.format('jpeg')))
        dataset = CustomDataset(im_names = index, label = None, transform = transform, 
                        im_path = None, feature = None, im_channel = im_channel)
    count1 = 0
    model.eval()    
    dataloader = DataLoader(dataset,
                                batch_size=32,
                                shuffle=False)
    with torch.no_grad():
        if is_sgsm:
            for img, _ in dataloader:
                img = img.to(device)
                output = model(img)
                _, predicted_label = torch.max(output, 1)
                total = predicted_label.sum().cpu().numpy()
                count1 += total
        else:
            for img, _ in dataloader:
                img = img.to(device)
                _, probas = model(img)
                _, predicted_label = torch.max(probas, 1)
                total = predicted_label.sum().cpu().numpy()
                count1 += total
    count0 = len(index) - count1
    count1 = count1/len(index)
    count0 = count0/len(index)
    return count1, count0


def mnist(req):
    feature1 = {
        'r1': '2',
        'r2': '3',
        'r3': '7',
        'r4': '9',
        'r5': '6',
        'r6': '0'
    }
    feature2 = {
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r3': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
        'r6': 'Very_Low_Height'
    }
    feature = {
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
    }
    mapping = {
        'r1': ['r1', 'r6'],
        'r2': ['r2', 'r3'],
        'r4': ['r4'],
        'r5': ['r5']
    }
    im_size = 64
    im_path = 'data/MNIST/train_images'
    df_loc = 'data/MNIST/train_data_for_reqs.csv'
    df = pd.read_csv(df_loc)
    df.set_index('images',inplace = True)
    transform = transforms.Compose([
                        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    lora_im_path = f'output/flux_lora_mnist_{req}/sample_gen'
    flux_im_path = f'output/flux_mnist_{req}/sample_gen'
    model_location = f'output/mnist_binary_classifier'
    im_index = find_index(df, feature1[req], feature2[req]) 

    return feature, im_path, im_index, transform, lora_im_path, flux_im_path, model_location, mapping
def celeba(req):
    feature1 = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat'
    }
    feature2 = 'Eyeglasses'
    im_path = 'data/CelebA-HQ/CelebA-HQ-img'
    df_loc = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno.txt'
    df = pd.read_csv(df_loc, skiprows = [0],sep = ' ')
    df.reset_index(level=1, drop=True, inplace=True)
    im_index = find_index(df, feature1[req], feature2) 
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    lora_im_path = f'output/flux_lora_celeba_{req}/sample_gen'
    flux_im_path = f'output/flux_celeba_{req}/sample_gen'
    model_location = f'output/celeba_binary_classifier_1024'
    return feature1, im_path, im_index, transform, lora_im_path, flux_im_path, model_location
def sgsm(req):
    im_path = f'data/SGSM/lora_data/{req}_data'
    lora_im_path = f'data/SGSM/all_generated_data/{req}'
    flux_im_path = f'data/SGSM/flux_sgsm_{req}'
    model_location = f'output/sgsm_binary_classifier'

    feature = {
        'f1': "f1",
        'f2': "f2",
        'f3a': "f3a",
        'f4a': "f4a",
        'f4b': "f4b",
        'f5a': "f5a",
        'f6': "f6",
        'f7': "f7"
    }
    im_index = None
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    return feature, im_path, im_index, transform, lora_im_path, flux_im_path, model_location
def calculate_JS(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for r in args.req:
        print(f'calculating JS metric for {r}')
        if args.dataset == 'mnist':
            feature, im_path, im_index, transform, lora_im_path, flux_im_path, model_location, mapping = mnist(r)
        elif args.dataset == 'celeba':
            feature, im_path, im_index, transform, lora_im_path, flux_im_path, model_location = celeba(r)
        elif args.dataset == 'sgsm':
            feature, im_path, im_index, transform, lora_im_path, flux_im_path, model_location = sgsm(r)
        pos_gtc = []
        neg_gtc = []
        pos_lora = []
        neg_lora = []
        pos_flux = []
        neg_flux = []
        fet = list(feature.keys())
        for f in fet:
            print(f'calculating for {feature[f]}')
            ##################calculate positive and negative value using GTCs
            ########calculate for real images
            model_name = model_location + f'_{f}/{feature[f].lower()}.pth'
            # print('model_name: ', model_name)
            if args.dataset == 'sgsm':
                model = load_model(model_name, device, is_sgsm = True)
            else:
                model = load_model(model_name, device)
            
            if args.dataset == 'mnist':
                countg1, countg0 = calculate_bin_using_GTC(im_path = im_path, model = model, transform = transform, device = device, im_channel = 1, im_index = im_index)
            elif args.dataset == 'sgsm':
                countg1, countg0 = calculate_bin_using_GTC(im_path = im_path, model = model, transform = transform, device = device, im_index = im_index, is_sgsm = True)
            else:
                countg1, countg0 = calculate_bin_using_GTC(im_path = im_path, model = model, transform = transform, device = device, im_index = im_index)
            pos_gtc.append(countg1)
            neg_gtc.append(countg0)
            print(f'for gtc, c1 = {countg1}, c0 = {countg0}')
            
            ####calculate for lora images
            if args.dataset == 'sgsm':
                countl1, countl0 = calculate_bin_using_GTC(im_path = lora_im_path, model = model, transform = transform, device = device, is_sgsm = True)
            else:
                countl1, countl0 = calculate_bin_using_GTC(im_path = lora_im_path, model = model, transform = transform, device = device)
            pos_lora.append(countl1)
            neg_lora.append(countl0)
            
            print(f'for lora, c1 = {countl1}, c0 = {countl0}')
            ####calculate for flux images
            if args.dataset == 'sgsm':
                countf1, countf0 = calculate_bin_using_GTC(im_path = flux_im_path, model = model, transform = transform, device = device, is_sgsm = True)
            else:
                countf1, countf0 = calculate_bin_using_GTC(im_path = flux_im_path, model = model, transform = transform, device = device)
            pos_flux.append(countf1)
            neg_flux.append(countf0)
            print(f'for flux, c1 = {countf1}, c0 = {countf0}')
            
        js_label_gtc = []
        js_real_lora = []
        js_real_flux = []
        for i in range(len(poslabel)):
            js_real_lora.append(distance.jensenshannon([pos_gtc[i], neg_gtc[i]], [pos_lora[i], neg_lora[i]], base = 2)**2)
            js_real_flux.append(distance.jensenshannon([pos_gtc[i], neg_gtc[i]], [pos_flux[i], neg_flux[i]], base = 2)**2)
            

        js_real_lora = np.mean(js_real_lora)
        js_real_flux = np.mean(js_real_flux)
        print(f'JS distance between the label and gtc: {js_label_gtc}, between real and lora image: {js_real_lora}, between real and flux image:{js_real_flux}\n\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq3')
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--req', nargs='+', default = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])

    args = parser.parse_args()
    
    calculate_JS(args)       