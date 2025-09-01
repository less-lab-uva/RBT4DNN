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
from gt_label_mnist import *

class MNISTDataset(Dataset):
    def __init__(self, split='train', im_path='data/MNIST', im_size = 64, im_channels=1):
        self.split = split
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        # self.save_location = save_location
        
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

    def __getitem__(self, index):
        ######image processing#####
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size, interpolation=T.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
            ])(im)

        ######text processing########
        digit = self.digits[index]        
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
        return im_tensor, digit, att_list


def make_data(save_location, split='train', 
                    im_path='data/MNIST', 
                    im_size = 64):
    '''
    For making a txt file with strings for all attributes
    '''
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    dataset = MNISTDataset(split=split, im_path=im_path, im_size = im_size)
    
    text = 'MNIST hand written digit in white on black background. '
    for index in tqdm(range(dataset.__len__())):
        img, digit, att_list = dataset.__getitem__(index)
        captionlist = []

        if digit == 8:
            captionlist.append(text + "The digit is an 8.")
        else:
            captionlist.append(text + f"The digit is a {digit}.")
        
        for j in range(len(att_list)):
            if j == 0 or j == 1 or j == 5:
                captionlist.append(f'The digit has {att_list[j]}.')
            else:
                captionlist.append(f'The digit is {att_list[j]}.')
        
        save_image(img,os.path.join(save_location, f'{index}.png'))
        with open(os.path.join(save_location,f'{index}.txt'), 'w') as f:
            f.write("\n".join(captionlist))

def make_customized_data_and(save_location, refdigit, attributes_index, attributes_text, split = "train", im_path='data/MNIST', 
                    im_size = 64):
    ''' For requirements with conjunctions
    Indices for attributes:
    0 = area,
    1 = length,
    2 = thickness,
    3 = slant,
    4 = width,
    5 = height
    '''
    if os.path.exists(save_location):
        print("removing old directory!!")
        os.system(f"rm -rf {save_location}")
    os.mkdir(save_location)
    dataset = MNISTDataset(split=split, im_path=im_path, im_size = im_size)
    
    for index in tqdm(range(dataset.__len__())):
        img, digit, att_list = dataset.__getitem__(index)
        if int(digit) != int(refdigit):
            continue
        text = ""
        if digit == 8:
            text = f'MNIST hand written digit in white on black background. The digit is an {digit}'
        else:
            text = f'MNIST hand written digit in white on black background. The digit is a {digit}'

        flag = 1
        hastext = []
        istext = []
        for j in range(len(attributes_index)):
            if attributes_text[j].lower().strip() == att_list[attributes_index[j]].lower().strip():
                if attributes_index[j] == 0 or attributes_index[j] == 1 or attributes_index[j] == 5:

                    hastext.append(f' {att_list[attributes_index[j]]}')                   
                else:
                    istext.append( f' {att_list[attributes_index[j]]}')
            else:    
                flag = 0
                
    
        if flag == 1: 
            if (len(istext) + len(hastext))>0:
                if (len(istext) + len(hastext)) == 1:
                    text = text + " and"
                else:
                    text = text + ","
                if len(istext)>0:
                    for i in range(len(istext)):
                        text = text + istext[i]
                        if i == len(istext)-2 and len(hastext)==0:
                            text = text + " and"
                        else:
                            text = text + ","
                if len(hastext)>0:
                    if len(hastext)==1 and len(istext)!= 0:
                        text = text + " and"
                    text = text + " has"
                    for i in range(len(hastext)):
                        text = text + hastext[i]
                        if i == len(hastext)-2:
                            text = text + " and"
                        elif i == len(hastext)-1:
                            text = text + "."
                        else:
                            text = text + "," 
            save_image(img,os.path.join(save_location, f'{index}.png'))
            with open(os.path.join(save_location,f'{index}.txt'), 'w') as f:
                f.write(text)
    lst = os.listdir(save_location)
    print("total number of images: ", len(lst)/2)

def make_customized_data_or(save_location, refdigit, attributes_index, attributes_text, split = "train", im_path='data/MNIST', 
                    im_size = 64):
    ''' For requirements with disjunctions
    Indices for attributes:
    0 = area,
    1 = length,
    2 = thickness,
    3 = slant,
    4 = width,
    5 = height
    '''
    if os.path.exists(save_location):
        print("removing old directory!!")
        os.system(f"rm -rf {save_location}")
    os.mkdir(save_location)
    dataset = MNISTDataset(split=split, im_path=im_path, im_size = im_size)
    
    # text = 'MNIST hand written digit in white on black background. '
    for index in tqdm(range(dataset.__len__())):
        img, digit, att_list = dataset.__getitem__(index)
        if int(digit) != int(refdigit):
            continue
        text = ""
        if digit == 8:
            text = f'MNIST hand written digit in white on black background. The digit is an {digit} and'
        else:
            text = f'MNIST hand written digit in white on black background. The digit is a {digit} and'

        flag = 0
        if attributes_index == 0 or attributes_index == 1 or attributes_index == 5:
            text = text + " has"
            
        else:
            text = text + " is"
        for j in range(len(attributes_text)):
            if attributes_text[j].lower().strip() == att_list[attributes_index].lower().strip():     
                flag = 1
                break
                
    
        if flag == 1: 
            for i in range(len(attributes_text)):
                text = text + " " + attributes_text[i]
                if i==len(attributes_text)-1:
                    text = text + "."
                else:
                    text = text + " or"

            save_image(img,os.path.join(save_location, f'{index}.png'))
            with open(os.path.join(save_location,f'{index}.txt'), 'w') as f:
                f.write(text)
    lst = os.listdir(save_location)
    print("total number of images: ", len(lst)/2)

if __name__ == "__main__":
    save_location = "data/mnist_r7"
    attributes_index = 3
    attributes_text = ["very thin", "very thick"]
    refdigit = 8
    make_customized_data_or(save_location = save_location, refdigit = refdigit, attributes_index = attributes_index, attributes_text = attributes_text)






    
    