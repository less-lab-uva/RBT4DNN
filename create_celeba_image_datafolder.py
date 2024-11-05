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

'''
This script creates requirement based image and text folder for LoRA training for CelebA-HQ dataset
'''


class CelebAHQ(Dataset):
    def __init__(self, split, im_path):
        self.split = split
        self.im_path = im_path
        self.img = []
        self.labels = []
        if 'train' in self.split:
            fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        
        if 'train' in self.split:
            f = os.path.join(im_path, "CelebAMask-HQ-attribute-anno.txt")
        self.df = pd.read_csv(f,skiprows = [0],sep = ' ')
        self.headers =self.df.columns.values.tolist()
        for header in self.headers:
            self.df.loc[self.df[header] == -1, header] = 0 

        for fname in tqdm(fnames):
            self.img.append(fname)
            im_name = os.path.split(fname)[1]
            self.labels.append(self.df.loc[[im_name]])
    
    def __len__(self):
        return len(self.img)
    def __getitem__(self,index):
        im = Image.open(self.img[index])
        return im, self.labels[index]




def make_requirement_based_dataset(dataset, attribute_1, attribute_2, text, 
                                        save_location, save_location_neg):
    ####create location to save
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    
    #### create the texts for lora training
    string = 'CelebAHQ close headshot of a person. '+ text
    print(string)
    labels = []
    ims = []
    
    
    dataset_len = dataset.__len__()
    for i in tqdm(range(dataset_len)):
        img, label = dataset.__getitem__(i)
        if attribute_2 is not None:
            if int(label[attribute_1].values) == 1 and int(label[attribute_2].values) == 1:
                ims.append(img)
        else:
            if int(label[attribute_1].values) == 1:
                ims.append(img)

    for i in tqdm(range(len(ims))):
        img = ims[i]
        img.save(os.path.join(save_location,f'{i}.png'))
        img.close()
        with open(os.path.join(save_location,f'{i}.txt'),'w') as f:
            f.write(string)


def dataset_making(req):
    dataset = CelebAHQ(split = 'train', im_path = 'data/CelebAMask-HQ/')
    text = {
        'r1': 'The person is wearing eyeglasses and has black hair.',
        'r2': 'The person is wearing eyeglasses and has brown hair.',
        'r3': 'The person is wearing eyeglasses and has a mustache.',
        'r4': 'The person is wearing eyeglasses and has wavy hair.',
        'r5': 'The person is wearing eyeglasses and is bald.',
        'r6': 'The person is wearing eyeglasses and a hat.'
    }
    attribute_1 = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat'
    }

    attribute_2 = 'Eyeglasses'
    save_location = f'data/celeba_{req}'
    make_requirement_based_dataset(dataset, attribute_1[req], attribute_2, 
                                    text[req], save_location, save_location_neg)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for celeba dataset filteration')
    parser.add_argument('--req', default='r1', type=str)
    args = parser.parse_args()

    dataset_making(args.req)
    
