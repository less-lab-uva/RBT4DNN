import glob
import os
import random
import torch
import glob
import torchvision
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import pandas as pd

class CelebAHQ(Dataset):
    def __init__(self, split, im_path, im_size =1024, im_channels = 3, vqa = False):
        self.split = split
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        self.img = []
        self.labels = []
        if 'train' in self.split:
            fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        
        if 'train' in self.split:
            if vqa:
                f = "data/CelebA-HQ/CelebA-HQ-MiniCPM-o-2_6-labels.csv"
                self.df = pd.read_csv(f, index_col = "index")
            else:
                f = os.path.join(im_path, "CelebAMask-HQ-attribute-anno.txt")
                self.df = pd.read_csv(f,skiprows = [0],sep = ' ')
        self.headers =self.df.columns.values.tolist()
        for header in self.headers:
            self.df.loc[self.df[header] == -1, header] = 0 

        for fname in tqdm(fnames):
            self.img.append(fname)
            im_name = os.path.split(fname)[1]
            if vqa:
                self.labels.append(self.df.loc[int(im_name.split(".")[0])])
            else:
                self.labels.append(self.df.loc[[im_name]])
    
    def __len__(self):
        return len(self.img)
    def __getitem__(self,index):
        im = Image.open(self.img[index])
        return im, self.labels[index], self.img[index]



def celeba_analysis():
    ###Find those pairs that have data between 100 and 300
    header = dataset.headers
    size = len(header)

    values = np.zeros([size, size],dtype = int)
    # print(values)
    total = np.zeros(size, dtype = int)
    for index in tqdm(range(dataset.__len__())):
        label = dataset.__getitem__(index)
        for i in range(size):
            if int(label[header[i]].values) == 1:
                total[i] += 1
            for j in range(size):
                if int(label[header[i]].values) == 1 and int(label[header[j]].values) == 1:
                    values[i][j]+=1
    
    np.set_printoptions(threshold = np.inf)
    print(values)

    np.savetxt('celeba-analysis.txt', values, fmt='%d')
    np.savetxt('celeba-total.txt',total, fmt = '%d')
    values = np.loadtxt('celeba-analysis.txt', dtype=int)
    total = np.loadtxt('celeba-total.txt', dtype=int)

    for i in range(size):
        for j in range(size):
            if values[i][j]>=100 and values[i][j]<=300:
                print(f'({header[i]}, {header[j]})= {values[i][j]}')
    print('\n\ntotal:')
    print(total)

def make_requirement_based_dataset(dataset,attribute_1, attribute_2, text, 
                                        save_location, save_location_neg, vqa = False):
    ####create location to save
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    
    if attribute_2 is None:
        if not os.path.exists(save_location_neg):
            os.mkdir(save_location_neg)
        neg_ims = []
    ####string for lora training
    if attribute_2 is not None:
        string = 'CelebAHQ close headshot of a person. '+ text
        # print(string)
        labels = []
    ims = []
    
    dataset_len = dataset.__len__()
    for i in tqdm(range(dataset_len)):
        img, label, img_index = dataset.__getitem__(i)
        flag = 0
        for att in attribute_1:
            if 'No_Beard' in att and ((vqa and label[att] == 0) or (not vqa and int(label[att].values) == 0)):
                flag = 1
                break
            elif 'No_Beard' not in att and ((vqa and label[att] == 1) or (not vqa and int(label[att].values) == 1)):
                flag = 1
                break

        if flag == 1:
            if attribute_2 is not None and ((vqa and label[attribute_2] == 1) or (not vqa and int(label[attribute_2].values) == 1)): 
                ims.append(img)
                labels.append(label)
            elif len(attribute_1) == 1: 
                if (vqa and label[attribute_1] == 1) or (not vqa and int(label[attribute_1].values) == 1):
                    ims.append(img)
                else:
                    neg_ims.append(img)

    if attribute_2 is None and len(ims)<len(neg_ims):
        neg_ims = random.sample(neg_ims, len(ims))
    for i in tqdm(range(len(ims))):
        img = ims[i]
        img.save(os.path.join(save_location,f'{i}.png'))
        img.close()
        if attribute_2 is not None:
            with open(os.path.join(save_location,f'{i}.txt'),'w') as f:
                f.write(string)
            labels[i].to_csv(os.path.join(save_location, f'{i}_label.txt'), sep=',', index=False)
        
        else:
            #############save neg images################
            img = neg_ims[i]
            img.save(os.path.join(save_location_neg,f'{i}.png'))
            img.close()

def dataset_making(req, is_single_attribute = False, vqa = False):
    dataset = CelebAHQ(split = 'train', im_path = '/project/lesslab/rbt4dnn/StableDiffusion-PyTorch/data/CelebAMask-HQ/', vqa = vqa)
    text = {
        'r1': 'The person is wearing eyeglasses and has black hair.',
        'r2': 'The person is wearing eyeglasses and has brown hair.',
        'r3': 'The person is wearing eyeglasses and has a mustache.',
        'r4': 'The person is wearing eyeglasses and has wavy hair.',
        'r5': 'The person is wearing eyeglasses and is bald.',
        'r6': 'The person is wearing eyeglasses and hat.',
        'r7': "The person is wearing eyeglasses and has a 5 o'clock shadow or goatee or mustache or beard or sideburns.",
        'post_condition': None
    }
    attribute_1 = {
        'r1': ['Black_Hair'],
        'r2': ['Brown_Hair'],
        'r3': ['Mustache'],
        'r4': ['Wavy_Hair'],
        'r5': ['Bald'],
        'r6': ['Wearing_Hat'],
        'r7': ['5_o_Clock_Shadow', 'Goatee','Mustache','No_Beard','Sideburns'],
        'post_condition': ['Eyeglasses']
    }
    if is_single_attribute:
        attribute_2 = None
        if vqa:
            save_location = f'data/celeba_minicpm_{attribute_1[req].lower()}'
            save_location_neg = f'data/celebahq_neg_minicpm_{attribute_1[req].lower()}'
        else:
            save_location = f'data/celeba_{attribute_1[req].lower()}'
            save_location_neg = f'data/celebahq_neg_{attribute_1[req].lower()}'
    else:
        attribute_2 = 'Eyeglasses'
        if vqa:
            save_location = f'data/celeba_minicpm_{req}'
        else:
            save_location = f'data/celeba_{req}'
        save_location_neg = None
    make_requirement_based_dataset(dataset, attribute_1[req], attribute_2, 
                                    text[req], save_location, save_location_neg, vqa = vqa)
    l = os.listdir(save_location)
    if attribute_2 is not None:
        num_sample = int(len(l)/3)
    else:
        num_sample = len(l)
    print(f'number of sample: {num_sample}')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for celeba dataset filteration')
    parser.add_argument('--req', default='r8', type=str)
    parser.add_argument('--is_single_attribute', default=False, type=bool)
    parser.add_argument('--is_vqa', default=False, type=bool)
    args = parser.parse_args()

    dataset_making(args.req, args.is_single_attribute, args.is_vqa)
    


