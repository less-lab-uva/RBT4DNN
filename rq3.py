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



def find_index(test_df, feature1, feature2 = None):
    ####Find index of the image that satisfies a requirement
    index = []
    for i in range(len(test_df.index)):
        if isinstance(feature1, list):
            flag = 0
            for f in feature1:
                if test_df[f].iloc[i] == 1:
                    flag = 1
                    break
            if flag == 1:
                if feature2 is None or test_df[feature2].iloc[i] == 1:
                    index.append(test_df.index[i])
                    
            
        else:
            if (feature2 is not None and test_df[feature1].iloc[i] == 1 and test_df[feature2].iloc[i] == 1) or (feature2 is None and test_df[feature1].iloc[i] == 1):
                index.append(test_df.index[i])
    return index
            

def load_model(model_name, device):
    model = ResnetModel().to(device)
    model.load_state_dict(torch.load(model_name, weights_only = True))
    model.eval()
    return model
def calculate_bin_using_GTC(im_path, model, transform, device, im_channel = 3, im_index = None):
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
                                batch_size=4,
                                shuffle=False)
    with torch.no_grad():
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
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r3': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
        'r6': 'Very_Low_Height',
        'r7': ['Very_Thin', 'Very_Thick'],
    }
    feature2 = {
        'r1': '2',
        'r2': '3',
        'r3': '7',
        'r4': '9',
        'r5': '6',
        'r6': '0',
        'r7': '8',
    }
    feature = {
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
        'r7a': 'Very_Thin'
    }
    im_size = 64
    im_path = 'data/MNIST/train_images'
    df_loc = 'data/MNIST/train.csv'
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

    return feature, im_path, im_index, df, transform, lora_im_path, flux_im_path, model_location
def celeba(req):
    feature1 = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat',
        'r7':['5_o_Clock_Shadow', 'Goatee', 'No_Beard', 'Mustache', 'Sideburns']
    }
    feature2 = 'Eyeglasses'
    feature = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat',
        'r7a':'5_o_Clock_Shadow', 
        'r7b': 'Goatee', 
        'r7c': 'No_Beard', 
        'r7d': 'Sideburns'
    }
    im_path = 'data/CelebA-HQ/CelebA-HQ-img'
    df_loc = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno.txt'
    df = pd.read_csv(df_loc, skiprows = [0],sep = ' ')
    df.reset_index(level=1, drop=True, inplace=True)
    df = pd.read_csv(df_loc)
    df.set_index("index", inplace = True)
    
    print("collecting image id")
    im_index = find_index(df, feature1[req], feature2) 
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    lora_im_path = f'output/flux_lora_celeba_{req}/sample_gen'
    lora_im_path_minicpm = f'output/flux_lora_celeba_minicpm_{req}/sample_gen'
    flux_im_path = f'output/flux_celeba_{req}/sample_gen'
    model_location = f'output/celeba_binary_classifier_1024'
    return feature, im_path, im_index, df, transform, lora_im_path, lora_im_path_minicpm, flux_im_path, model_location
def calculate_JS(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    js_real_flux_all = []
    js_real_lora_all = []
    js_real_lora_minicpm_all = []
    for r in args.req:
        print(f'calculating JS metric for {r}')
        if args.dataset == 'mnist':
            feature, im_path, im_index, df, transform, lora_im_path, flux_im_path, model_location = mnist(r)
        elif args.dataset == 'celeba':
            feature, im_path, im_index, df, transform, lora_im_path, lora_im_path_minicpm, flux_im_path, model_location = celeba(r)
        poslabel = []
        neglabel = []
        pos_gtc = []
        neg_gtc = []
        pos_lora = []
        pos_lora_minicpm = []
        neg_lora = []
        neg_lora_minicpm = []
        pos_flux = []
        neg_flux = []
        
        pos_entropy = []
        neg_entropy = []
        fet = list(feature.keys())
        for f in fet:
            print(f'f: {f}: calculating for {feature[f]}')
            ######calculate positive and negative value from label#######
            count1 = 0
            count0 = 0
            for i in im_index:
                if df[feature[f]].loc[i] == 1:
                    count1 += 1
                else:
                    count0 += 1
            poslabel.append(count1/len(im_index))
            neglabel.append(count0/len(im_index))
            ##################calculate positive and negative value using GTCs
            ########calculate for real images
            if 'r7' in f:
                model_name = model_location + f'_{f[:-1]}/{feature[f].lower()}.pth'
            else:
                model_name = model_location + f'_{f}/{feature[f].lower()}.pth'
            model = load_model(model_name, device)
            
            if args.dataset == 'mnist':
                countg1, countg0 = calculate_bin_using_GTC(im_path = im_path, model = model, transform = transform, device = device, im_channel = 1, im_index = im_index)
            else:
                countg1, countg0 = calculate_bin_using_GTC(im_path = im_path, model = model, transform = transform, device = device, im_index = im_index)
            pos_gtc.append(countg1)
            neg_gtc.append(countg0)
            print(f'for gtc, c1 = {countg1}, c0 = {countg0}')
            
            ####calculate for lora images
            countl1, countl0 = calculate_bin_using_GTC(im_path = lora_im_path, model = model, transform = transform, device = device)
            pos_lora.append(countl1)
            neg_lora.append(countl0)
            print(f'for lora, c1 = {countl1}, c0 = {countl0}')
            if args.dataset == 'celeba':
                countl1, countl0 = calculate_bin_using_GTC(im_path = lora_im_path_minicpm, model = model, transform = transform, device = device)
                pos_lora_minicpm.append(countl1)
                neg_lora_minicpm.append(countl0)
                print(f'for lora and VQA, c1 = {countl1}, c0 = {countl0}')
            
            
            ####calculate for flux images
            countf1, countf0 = calculate_bin_using_GTC(im_path = flux_im_path, model = model, transform = transform, device = device)
            pos_flux.append(countf1)
            neg_flux.append(countf0)
            print(f'for flux, c1 = {countf1}, c0 = {countf0}')
            
        js_label_gtc = []
        js_real_lora = []
        js_real_lora_minicpm =[]
        js_real_flux = []
        entropy_lora = []
        for i in range(len(poslabel)):
            # js_label_gtc.append(distance.jensenshannon([poslabel[i], neglabel[i]], [pos_gtc[i], neg_gtc[i]], base = 2)**2)
            js_real_lora.append(distance.jensenshannon([pos_gtc[i], neg_gtc[i]], [pos_lora[i], neg_lora[i]], base = 2)**2)
            js_real_flux.append(distance.jensenshannon([pos_gtc[i], neg_gtc[i]], [pos_flux[i], neg_flux[i]], base = 2)**2)
            if args.dataset == 'celeba':
                js_real_lora_minicpm.append(distance.jensenshannon([pos_gtc[i], neg_gtc[i]], [pos_lora_minicpm[i], neg_lora_minicpm[i]], base = 2)**2)
            
        js_real_lora = np.mean(js_real_lora)
        if args.dataset == 'celeba':
            js_real_lora_minicpm = np.mean(js_real_lora_minicpm)
        js_real_flux = np.mean(js_real_flux)
        js_real_flux_all.append(round(js_real_flux, 5))
        js_real_lora_all.append(round(js_real_lora, 5))
        if args.dataset == 'celeba':
            print(f'JS distance between real and lora image: {js_real_lora}, between real and VQA lora image: {js_real_lora_minicpm}, between real and flux image:{js_real_flux}\n\n')
            js_real_lora_minicpm_all.append(round(js_real_lora_minicpm, 5))
        else:
            print(f'JS distance between real and lora image: {js_real_lora}, between real and flux image:{js_real_flux}\n\n')
    print("\n\nlora: ",js_real_lora_all)
    if args.dataset == 'celeba':
        print("lora VQA: ",js_real_lora_minicpm_all)
    print("flux: ",js_real_flux_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq3')
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--req', nargs='+', default = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7'])


    args = parser.parse_args()
    
    calculate_JS(args)       