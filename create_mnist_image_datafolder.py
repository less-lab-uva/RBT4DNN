import glob
import os
import random
from PIL import Image
import pandas as pd 
from tqdm import tqdm
import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as T
from mnist_text import *

class MNISTDataset(Dataset):
    def __init__(self, split='train', im_path='data/MNIST', im_size = 64, im_channels=1):
        self.split = split
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        self.save_location = save_location
        
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)

        lst = os.listdir(os.path.join(im_path, '{}_images/'.format(self.split))) # your directory path
        self.len = len(lst)
        ##########save imagefile name###########
        self.images = []
        for i in range(self.len):
            fname = os.path.join(im_path, '{}_images/{}.{}'.format(self.split,i,'png'))
            self.images.append(fname)
        ##########save morpho information######
        self.df = pd.read_csv(os.path.join(im_path, f"{self.split}-morpho.csv"))

        #############save digit#########
        f = open(os.path.join(im_path, f"MNIST_{self.split}_label.txt"), "r")
        self.digits = []
        for line in f.readlines():
            self.digits.append(int(line))
        f.close() 
    
    def __len__(self):
        return self.len

    def __getitem__(self, index,attribute = None,temp = 0):
        ######image processing#####
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size, interpolation=T.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
            ])(im)

        ######text processing########
        digit = self.digits[index]
        attr = {
            'area': area,
            'length': length,
            'thickness': thickness,
            'slant': slant,
            'width': width,
            'height': height
        }.get(attribute)
        ##to indicate area or wide
        if attribute is not None:
            if temp ==1:            
                string = attr(self.df, index, digit)
            else:
                string = attr(self.df, index)
            att_list = []
            l = area(self.df, index, digit)
            att_list.append(l)
            l = length(self.df, index)
            att_list.append(l)
            l = thickness(self.df, index)
            att_list.append(l)
            l = slant(self.df, index)
            att_list.append(l)
            l = width(self.df, index, digit)
            att_list.append(l)
            l = height(self.df, index)
            att_list.append(l)
        else:
            string = None
            att_list = []
            
        
        return im_tensor, digit, string, att_list


        

def make_data(digit, attributetext, attribute, save_location, attribute_2 = None, split='train', 
                    im_path='data/MNIST', im_size = 64):
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    dataset = MNISTDataset(split=split, im_path=im_path, im_size = im_size)
    if digit == 8:
        text = f'MNIST hand written digit in white on black background. The digit is an {digit} and'
    else:
        text = f'MNIST hand written digit in white on black background. The digit is a {digit} and'
    if 'area' in attribute or 'length' in attribute or 'height' in attribute:
        text = text + f' has {attributetext}.'
    else:
        text = text + f' is {attributetext}.'
    count = 0
    pos_img = []
    neg_img = []
    label_list = []
    for index in tqdm(range(dataset.__len__())):
        if 'area' in attribute or 'width' in attribute:
            img, digitvalue, string, label = dataset.__getitem__(index,attribute,temp = 1)
        else:
            img, digitvalue, string, label = dataset.__getitem__(index,attribute,temp = 0)

        if digit == int(digitvalue) and attributetext == string:
            pos_img.append(img)
       
    for i in tqdm(range(len(pos_img))):
        save_image(pos_img[i],os.path.join(save_location, f'{i}.png'))
        with open(os.path.join(save_location,f'{i}.txt'), 'w') as f:
            f.write(text)
    


if __name__=='__main__':
    reqs = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    text = {
        'r1': 'very low height',
        'r2': 'very thick',
        'r3': 'very thick',
        'r4': 'very left leaning',
        'r5': 'very right leaning',
        'r6': 'very low height'
    }
    
    img_path = 'data/MNIST'
    im_size = 64
    digit ={
        'r1': 2,
        'r2': 3,
        'r3': 7,
        'r4': 9,
        'r5': 6,
        'r6': 0,
    }
    
    attribute = {
        'r1': 'height',
        'r2': 'thickness',
        'r3': 'thickness',
        'r4': 'slant',
        'r5': 'slant',
        'r6': 'height'
    }
    
    for req in reqs:    
        save_location=f'data/mnist_{req}'
        make_data(digit = digit[req], attributetext = text[req], attribute = attribute[req], save_location = save_location,
                    split='train', im_path=img_path, im_size = im_size)
    
    