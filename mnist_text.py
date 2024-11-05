import torch
import os
import torchvision
import pandas as pd 
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms

def area(df,index, label):
    x = df["area"].iloc[index]
    if label == 1: #very_small_area(x) = x.area<30, small_area(x) = 30<= x.area<50, large_area(x) = 50<=x.area<100, very_large_area(x) = x.area>=100
        if x<30:
            return "very small area"
        elif x<50:
            return "small area"
        elif x<100:
            return "large area"
        else:
            return "very large area"
    else: #very_small_area(x) = x.area<30, small_area(x) = 30<= x.area<100, large_area(x) = 100<=x.area<300, very_large_area(x) = x.area>=300
        if x<30:
            return "very small area"
        elif x<100:
            return "small area"
        elif x<300:
            return "large area"
        else:
            return "very large area"

def length(df,index):
    x = df["length"].iloc[index] #extreme_short(x) = x.length<10 , short(x) = 10<=x.length<35 , long(x) = 35<=x.length<50 , extreme_long(x) = x.length>50
    if x <10:
        return "very short length"
    elif x<35:
        return "short length"
    elif x<50:
        return "long length"
    else:
        return "very long length"

def thickness(df,index):
    x = df["thickness"].iloc[index] #extreme_thin(x) = x.thickness<=1.5, thin(x) = 1.5<x.thickness<3.5, slightly_thick(x) = 3.5<=x.thickness<4, thick(x) = 4<=x.thickness<7, extreme_thick(x) = x.thickness>=7
    if x<=1.5:
        return "very thin"
    elif x<3.5:
        return "thin"
    elif x<4:
        return "slightly thick"
    elif x<7:
        return "thick"
    else:
        return "very thick"

def slant(df,index):
    x = df["slant"].iloc[index] #upright(x) = |x.slant|<=0.1, left(x) = -0.4<x.slant<-0.1, right(x) = 0.1<x.slant<0.4, extreme_left(x) = x.slant<=-0.4, extreme_right(x) = x.slant>=0.4
    if x>=-0.1 and x<=0.1:
        return "upright"
    elif x>-0.4 and x<-0.1:
        return "left leaning"
    elif x>0.1 and x<0.4:
        return "right leaning"
    elif x<=-0.4:
        return "very left leaning"
    elif x>=0.4:
        return "very right leaning"

def width(df,index, label):
    x = df["width"].iloc[index] 
    if label ==1: #extreme_narrow(x) = x.width<3.5, narrow(x) = 3.5<=x.width<5, wide(x) = 5<=x.width<7.5, extreme_wide(x) = x.width>=7.5
        if x<3.5:
            return "very narrow"
        elif x<5:
            return "narrow"
        elif x<7.5:
            return "wide"
        else:
            return "very wide"
    
    else:#extreme_narrow(x) = x.width<9, narrow(x) = 9<=x.width<12, wide(x) = 12<=x.width<18, extreme_wide(x) = x.width>=18
        if x<9:
            return "very narrow"
        elif x<12:
            return "narrow"
        elif x<18:
            return "wide"
        else:
            return "very wide"

def height(df,index):
    x = df["height"].iloc[index] #x.height<14, low(x) = 14<=x.height<17, high(x) = 17<= x.height(x)<20, extreme_high(x) = x.height>=20     
    if x<14:
        return "very low height"
    elif x<17:
        return "low height"
    elif x<20:
        return "high height"
    else:
        return "very high height"

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

class MNISTDataset(Dataset):
    def __init__(self, split='train', im_path='data/MNIST', im_size = 64, im_channels=1):
        self.split = split
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        self.save_location = save_location
        
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)

        lst = os.listdir(os.path.join(im_path, '{}_images_ori/'.format(self.split))) # your directory path
        self.len = len(lst)
        ##########save imagefile name###########
        self.images = []
        for i in range(self.len):
            fname = os.path.join(im_path, '{}_images_ori/{}.{}'.format(self.split,i,'png'))
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
                torchvision.transforms.Resize(self.im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ])(im)
        im.close()

        ######text processing########
        digit = self.digits[index]

        Area = area(self.df,index, digit)
        Length = length(self.df,index)
        Thickness = thickness(self.df,index)
        Slant = slant(self.df,index)
        Width = width(self.df,index, digit)
        Height = height(self.df,index)

        if digit == 8:
            string = f"MNIST hand written digit in white on black background. The digit is an {digit}, {Thickness}, {Slant}, {Width}, has {Area}, {Length} and {Height}."
        else:
            string = f"MNIST hand written digit in white on black background. The digit is a {digit}, {Thickness}, {Slant}, {Width}, has {Area}, {Length} and {Height}."
        return im_tensor, string

def make_data_for_entire_set():
    save_location = 'data/MNIST/train_images'
    if not os.path.exists(save_location):
        os.mkdir(save_location)
    dataset = MNISTDataset()
    for index in tqdm(range(dataset.__len__())):
        img,text = dataset.__getitem__(index)
        save_image(img,os.path.join(save_location, f'{index}.png'))
        with open(os.path.join(save_location,f'{index}.txt'), 'w') as f:
            f.write(text)

def make_data_with_requirements():
    save_location = 'data/MNIST/train_images_req'
    if not os.path.exists(save_location):
        os.mkdir(save_location)
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
    index = []
    df_loc = 'data/MNIST/train_data_for_reqs.csv'
    df = pd.read_csv(df_loc)
    df.set_index('images',inplace = True)

    
    req = list(feature1.keys())
    for r in req:
        ind = find_index(df, feature1[r], feature2[r])
        for i in ind:
            index.append(i)
    img_path = 'data/MNIST/train_images'
    for i in index:
        print(i)
        img = Image.open(os.path.join(img_path,i))
        img.save(os.path.join(save_location, i))
        img.close()
        with open(os.path.join(img_path, i.replace('png','txt')), 'r') as f:
            text = f.read()
        print(text)
        with open(os.path.join(save_location, i.replace('png','txt')),'w') as f:
            f.write(text)


if __name__=='__main__':
    make_data_with_requirements()

    





