import os
import time
import glob
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

class MyVitModel(nn.Module):
    def __init__(self):
        super(MyVitModel, self).__init__()
        self.model_name = "vit-patch16-384"

        # Instantiate Vit_b_16
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-384")

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=self.vit.pooler.dense.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=2),
        )

    def forward(self, x):
        # output = self.vit(x, interpolate_pos_encoding = True)
        output = self.vit(x)
        # Take the mean of the sequence length dimension (dim=1) from the last_hidden_state
        # https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281/2
        features = output.last_hidden_state.mean(dim=1)
        logits = self.classifier_head(features)
        probas = F.softmax(logits, dim = 1)
        return logits, probas
class ResnetModel(nn.Module):
    def __init__(self, num_of_class = 2, is_sgsm = False):
        super(ResnetModel, self).__init__()
        self.is_sgsm = is_sgsm
        resnet = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=resnet.fc.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=num_of_class),
        )

    def forward(self, x):
        features = self.resnet(x)
        logits = self.classifier_head(features)
        if self.is_sgsm:
            return logits
        probas = F.softmax(logits, dim=1)
        return logits, probas
class CustomDataset(Dataset):
    def __init__(self, im_names, label, transform, im_path = None, feature = None, im_channel = 3):
        '''
        im_names = names of the images to make the dataset
        im_path = path to the image folder,
        label = pandas dataframe with the values for all features
        transform = transform function to use for images
        feature = feature to consider, if None, return all
        '''   
        self.im = im_names
        self.im_path = im_path
        self.label = label
        self.transform = transform
        self.feature = feature
        self.im_channel = im_channel
    def __len__(self):
        return len(self.im)
    def __getitem__(self,index):
        if self.im_path is not None:
            im_name = os.path.join(self.im_path, self.im[index])
        else:
            im_name = self.im[index]
        if self.im_channel == 1:
            img = Image.open(im_name).convert('RGB')
        else:
            img = Image.open(im_name)
        im_tensor = self.transform(img)  
        img.close()
        if self.label is not None:
            if self.feature is not None:
                labelvalue = self.label[self.feature].loc[self.im[index]]
            else:
                labelvalue = self.label.loc[self.im[index]].to_dict()
        else:
            labelvalue = 1
        return im_tensor, [labelvalue, self.im[index]]  


def compute_accuracy(probas, labels):
    _, predicted_labels = torch.max(probas, 1)
    correct_pred = (predicted_labels == labels).sum().item()
    return correct_pred/labels.size(0)
    

def train_epoch(dataloader, model, cost_fn, optimizer, device):
    model.train()
    train_acc = 0
    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        cost = cost_fn(logits, targets)
        acc = compute_accuracy(probas, targets)
        train_acc +=acc
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if batch_idx % 50 == 0:
            print ('Batch %04d/%04d | Cost: %.4f' 
                %(batch_idx, 
                len(dataloader), cost))
    print(f'Training accuracy: {train_acc/len(dataloader)}')

    return model
def val_epoch(dataloader, model, cost_fn, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            logits, probas = model(features)
            cost = cost_fn(logits, targets)
            acc = compute_accuracy(probas, targets)

            val_loss += cost.item()
            val_acc += acc
        
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc

def train(train_loader, val_loader, test_loader,
                learning_rate, epochs, save_location, model, model_name, num_of_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists(save_location):
        os.mkdir(save_location)
    
    #####Configure training parameters############
    cost_fn = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ###training starts##########
    training_time = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                    threshold=1e-3, factor=0.5, patience=2, verbose=True)
    
    for epoch in range(epochs):
        print(f'starting epoch {epoch+1}')
        start_time = time.time()
        model = train_epoch(dataloader = train_loader, model = model,
                                    cost_fn = cost_fn, 
                                    optimizer = optimizer,
                                    device = device)

        req_time = time.time() - start_time
        training_time = training_time + req_time
        print('Time taken for the epoch training: %.2f min' % ((req_time)/60))

        #########validdation##########
        val_loss, val_acc = val_epoch(dataloader = val_loader, model= model,
                                     cost_fn = cost_fn, device = device)
        print(f'validation accuracy: {val_acc}, validation_loss = {val_loss}')

        ###########model_save##########
        model.eval()
        if num_of_classes == 2:
            torch.save(model.state_dict(), os.path.join(save_location,model_name))
            
        else:
            torch.save(model, os.path.join(save_location,model_name))
        
        # Step the learning rate scheduler based on validation loss
        # scheduler.step(val_loss)

        # Early stopping
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-6:
            print(f"Early stopping at epoch {epoch+1}. Learning rate has reduced below the threshold.")
            break
    print("training finished.....")
    print('calculating test accuracy.......')
    test_loss, test_acc = val_epoch(dataloader = test_loader, model= model,
                                     cost_fn = cost_fn, device = device)
    print(f'test accuracy: {test_acc}, validation_loss = {test_loss}')
    
    
    

def split_train_valid(train_df, feature_name, vratio):
    ##############compute balanced pos and neg for a feature the index of the dataframe is image name##########
    pos = []
    neg = []
    for i in tqdm(range(len(train_df.index))):
        if train_df[feature_name].iloc[i] == 1:
            pos.append(train_df.index[i])
        else:
            neg.append(train_df.index[i])

    print("total number of data: ", len(train_df.index))
    print('\ntotal number of pos data for ', feature_name,": ", len(pos))
    print('total number of neg data for ', feature_name,": ", len(neg))

    if len(pos) > len(neg):
        pos = random.sample(pos, len(neg))
    elif len(pos) < len(neg):
        neg = random.sample(neg, len(pos))
    
    pcount = 0
    ncount = 0
    for i in tqdm(range(len(pos))):
        pcount += train_df[feature_name].loc[pos[i]]
        if train_df[feature_name].loc[neg[i]] == 0:
            ncount += 1 
    print('\ntotal number of pos data for ', feature_name," in balanced data : ",pcount)
    print('total number of neg data for ', feature_name," in balanced data : ",ncount)

    ####split into train, validate
    if vratio != 0:
        size = len(pos)
        v_size = int(size * vratio)
        print("vsize: ", v_size)
        valid = random.sample(pos, v_size)
        valid_neg = random.sample(neg, v_size)

        for d in valid:
            pos.remove(d)
        for d in valid_neg:
            neg.remove(d)
        train = pos
        print("\ntotal train pos: ",len(train))
        print('total train neg: ', len(neg))
        print('total valid pos: ', len(valid))
        print(' total valid neg: ', len(valid_neg))

        for d in neg:
            train.append(d)
        for d in valid_neg:
            valid.append(d)
        print('total train: ', len(train))
        print('total valid: ',len(valid))
    else:
        train = pos
        for d in neg:
            train.append(d)
        print('total data: ', len(train))
        valid = None
    return train, valid


def celeba_classifier(req, model_type = 'resnet', vratio = 0.1):
    feature_name = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat',
        'LC_384': 'Eyeglasses'
    }
    im_size = 384
    batch_size = 8
    learning_rate = 1e-3
    epochs = 50
    save_location = f'output/celeba_binary_classifier_1024_{req}'
    model_name = f'{feature_name[req].lower()}.pth'
    num_of_classes = 2

    #####make training, validation and test dataset
    im_path = 'data/CelebA-HQ/CelebA-HQ-img'
    train_label_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-train.csv'
    test_label_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-test.csv'
    train_df = pd.read_csv(train_label_path)
    train_df = train_df.set_index('images')
    test_df = pd.read_csv(test_label_path)
    test_df = test_df.set_index('images')

    ###########split train df into train and validadtion set###############
    train_set, valid_set = split_train_valid(train_df, feature_name[req], vratio)
    test_set, _ = split_train_valid(test_df, feature_name[req], 0)

    
    #############configure model and transform function###############
    if 'resnet' in model_type:
        model = ResnetModel(num_of_class = num_of_classes)
        transform = transforms.Compose([
                        # transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    elif 'vit' in model_type:
        model = MyVitModel()
        transform = transforms.Compose([
                        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                    ])

    ############make dataset################
    train_dataset = CustomDataset(im_names = train_set, label = train_df, 
            transform = transform, im_path = im_path, feature = feature_name[req])
    valid_dataset = CustomDataset(im_names = valid_set, label = train_df, 
            transform = transform, im_path = im_path, feature = feature_name[req])
    test_dataset = CustomDataset(im_names = test_set, label = test_df, 
            transform = transform, im_path = im_path, feature = feature_name[req])

    ############make dataloader################
    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    ##############train and save model###############
    print('Training is starting.....')
    train(train_loader, val_loader, test_loader, learning_rate, epochs, save_location, model, model_name, num_of_classes)


def mnist_classifier(req, model_type = 'resnet', vratio = 0.1):
    feature_name = {
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
        '2' : 'r1',
        '3' : 'r2',
        '7' : 'r3',
        '9' : 'r4',
        '6' : 'r5',
        '0' : 'r6'
    }
    im_size = 64
    batch_size = 32
    learning_rate = 1e-3
    epochs = 100
    if req == '2' or req == '3' or req == '7' or req == '9' or req == '6' or req == '0':
        req_name = feature_name[req]
        fe_name = req
    else:
        req_name = req
        fe_name = feature_name[req]
    save_location = f'output/mnist_binary_classifier_{req_name}'
    model_name = f'{fe_name.lower()}.pth'
    # print("location to save: ",os.path.join(save_location,model_name))
    num_of_classes = 2
    #####make training, validation and test dataset
    im_path = 'data/MNIST/train_images'
    train_label_path = 'data/MNIST/train.csv'
    test_label_path = 'data/MNIST/test.csv'
    train_df = pd.read_csv(train_label_path)
    train_df = train_df.set_index('images')
    test_df = pd.read_csv(test_label_path)
    test_df = test_df.set_index('images')
    # print('test index:')
    # print(test_df.index)

    ###########split train df into train and validadtion set###############
    train_set, valid_set = split_train_valid(train_df, fe_name, vratio)
    test_set, _ = split_train_valid(test_df, fe_name, 0)

    
    #############configure model and transform function###############
    if 'resnet' in model_type:
        model = ResnetModel(num_of_class = num_of_classes)
        transform = transforms.Compose([
                        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    ############make dataset################
    train_dataset = CustomDataset(im_names = train_set, label = train_df, 
            transform = transform, im_path = im_path, feature = fe_name, im_channel = 1)
    valid_dataset = CustomDataset(im_names = valid_set, label = train_df, 
            transform = transform, im_path = im_path, feature = fe_name, im_channel = 1)
    test_dataset = CustomDataset(im_names = test_set, label = test_df, 
            transform = transform, im_path = im_path, feature = fe_name, im_channel = 1)

    ############make dataloader################
    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    ##############train and save model###############
    print('Training is starting.....')
    train(train_loader, val_loader, test_loader, learning_rate, epochs, save_location, model, model_name, num_of_classes)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for classifier training')
    parser.add_argument('--fet', default='model_under_test', type=str)
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--classifier', default='binary', type=str)
    parser.add_argument('--model_type', default='resnet', type=str)
    parser.add_argument('--validratio', default=0.1, type=float)
    args = parser.parse_args()
    if args.dataset == 'celeba':
        celeba_classifier(args.fet, args.model_type, args.validratio)
    elif args.dataset == 'mnist':
        if args.classifier == 'binary':
            mnist_classifier(args.fet, args.model_type, args.validratio)
        elif args.classifier == 'multi':
            mnist_multiclassifier()