import os
import torch
import torch.nn as nn
import torchvision
import argparse
import random
from tqdm import tqdm
import pandas as pd
import json
from torch.utils.data import DataLoader
from diffusers import FluxPipeline

from train_classifier import ResnetModel, CustomDataset
from PIL import Image
from torchvision import transforms


def find_r_data(test_df, feature1, feature2 = None):
    ##############compute balanced pos and neg for a feature the index of the dataframe is image name##########
    pos = []
    neg = []
    for i in range(len(test_df.index)):
        if feature2 is not None and test_df[feature1].iloc[i] == 1 and test_df[feature2].iloc[i] == 1:
            pos.append(test_df.index[i])
        elif feature2 is None and test_df[feature1].iloc[i] == 1:
            pos.append(test_df.index[i])
        else:
            neg.append(test_df.index[i])

    print("\ntotal number of data: ", len(test_df.index))
    print('total number of pos data for ', feature1," and ", feature2, ": ", len(pos))

    return pos

def compute_match(dataloader, model1, model2, device, feature1 = None, feature2 = None):
    sample_size = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    mismatch = []
    img_list = []
    pred1 = []
    pred2 = []
    labels1 = []
    labels2 = []
    with torch.no_grad():
        for images, data in dataloader:
            images = images.to(device)
            labels, img_index = data
            if feature1 is not None:
                label1 = labels[feature1]
            else:
                label1 = torch.ones(images.shape[0])
            labels1.extend(label1.tolist())
            label1 = label1.to(device)

            if feature2 is not None:
                label2 = labels[feature2]
            else:
                label2 = torch.ones(images.shape[0])
            labels2.extend(label2.tolist())
            label2 = label2.to(device)

            _, probas1 = model1(images)
            _, predicted_labels1 = torch.max(probas1, 1)
            if model2 is not None:
                _, probas2 = model2(images)
                _, predicted_labels2 = torch.max(probas2, 1)
                pred2.extend(predicted_labels2.tolist())
            
            ######add all to results#######
            img_list.extend(img_index)
            pred1.extend(predicted_labels1.tolist())   

            if model2 is not None:
                count = 0
                for i in range(len(label1)):
                    if predicted_labels1[i] == label1[i] and predicted_labels2[i] == label2[i]:
                        correct = correct+1
                        # mismatch.append(img_index[i])
                    else:
                        mismatch.append(img_index[i])
            else:
                for i in range(len(label1)):
                    if predicted_labels1[i] == label1[i]:
                        correct = correct+1
                    else:
                        mismatch.append(img_index[i])

            sample_size = sample_size + label1.size(0)
    print(f'Total number of sample: {sample_size}')
    print(f'Total number of match: {correct}')

    return correct/sample_size, mismatch, img_list, pred1, pred2, labels1, labels2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mnist(req, flag = True, turn = None, datatype = None):
    im_size = 64
    batch_size = 64
    transform = transforms.Compose([
                        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    if flag:
        test_label_path = 'data/MNIST/test.csv'
        test_df = pd.read_csv(test_label_path)
        test_df = test_df.set_index('images')
        im_path = 'data/MNIST/train_images'
        # print("im_names: ")
        # print(im_names)
        feature_name = {
            'r1': 'Very_Low_Height',
            'r2': 'Very_Thick',
            'r3': 'Very_Thick',
            'r4': 'Very_Left_Leaning',
            'r5': 'Very_Right_Leaning',
            'r6': 'Very_Low_Height'
        }
        digit = {
        'r1': '2',
        'r2': '3',
        'r3': '7',
        'r4': '9',
        'r5': '6',
        'r6': '0'
        }
        feature1 = feature_name[req]
        feature2 = digit[req]
        im_names = find_r_data(test_df, feature1, feature2)
        ori_dataset = CustomDataset(im_names = im_names, label = test_df, transform = transform, 
                    im_path = im_path, feature = None, im_channel = 1)

        ori_dataloader = DataLoader(ori_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
        gen_dataloader = None

        
        model1_location = f'output/mnist_binary_classifier_{req}/{feature_name[req].lower()}.pth'
        model1 = ResnetModel().to(device)
        model1.load_state_dict(torch.load(model1_location, weights_only = True))

        model2_location = f'output/mnist_binary_classifier_{req}/{digit[req]}.pth'
        model2 = ResnetModel().to(device)
        model2.load_state_dict(torch.load(model2_location, weights_only = True))

        model1.eval()
        model2.eval()
        
    else:
        start = turn * 100
        end = turn * 100 + 100
        im_names = [f'{i}.png' for i in range(start, end)]
        if datatype == "alldata":
            im_path = 'data/images_from_loras/Alldata_M' + req.split('r')[1]
        elif datatype == "allreq":
            im_path = 'data/images_from_loras/Allreq_M' + req.split('r')[1]
        else:
            im_path = 'data/images_from_loras/M' + req.split('r')[1]
        gen_dataset = CustomDataset(im_names = im_names, label = None, transform = transform, 
                    im_path = im_path, feature = None, im_channel = 1)              
        gen_dataloader = DataLoader(gen_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        ori_dataloader = None
        model1 = None
        model2 = None
        feature1 = None
        feature2 = None
    
    return ori_dataloader, gen_dataloader, model1, model2, feature1, feature2

def celeba(req, flag = True, turn = None):
    im_size = 1024
    batch_size = 32
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    feature_name = {
        'r1':'Black_Hair',
        'r2':'Brown_Hair',
        'r3':'Mustache',
        'r4':'Wavy_Hair',
        'r5':'Bald',
        'r6':'Wearing_Hat'
    }
    model1_location = f'output/celeba_binary_classifier_1024_{req}/{feature_name[req].lower()}.pth'
    print('model1: ', model1_location)
    
    model2_location = f'output/celeba_binary_classifier_1024_{req}/eyeglasses.pth'
    print('model2: ', model2_location)

    if flag:
        im_path = 'data/CelebA-HQ/CelebA-HQ-img'
        test_label_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-test.csv'
        test_df = pd.read_csv(test_label_path)
        test_df = test_df.set_index('images')

        feature1 = feature_name[req]
        feature2 = 'Eyeglasses'
        im_names = find_r_data(test_df, feature1, feature2)
        ori_dataset = CustomDataset(im_names = im_names, label = test_df, transform = transform, 
                im_path = im_path, feature = None, im_channel = 3)
        ori_dataloader = DataLoader(ori_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

        model1 = ResnetModel().to(device)
        model1.load_state_dict(torch.load(model1_location, weights_only = True))
        
        model2 = ResnetModel().to(device)
        model2.load_state_dict(torch.load(model2_location, weights_only = True))
        model1.eval()
        model2.eval()
        gen_dataloader = None
    else:
        start = turn * 100
        end = turn * 100 + 100
        im_names = [f'{i}.png' for i in range(start, end)]
        im_path = 'data/images_from_loras/C' + req.split('r')[1]
        gen_dataset = CustomDataset(im_names = im_names, label = None, transform = transform, 
                    im_path = im_path, feature = None, im_channel = 1)                
        
        gen_dataloader = DataLoader(gen_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        ori_dataloader = None
        # ori_dataset = None
        model1 = None
        model2 = None
        feature1 = None
        feature2 = None
        
    

    return ori_dataloader, gen_dataloader, model1, model2, feature1, feature2

def main(args):  
    print('calculating for ', args.dataset)
    req = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    dataset = args.dataset
    num_of_samples = args.num_samples 
    num_of_samples_per_epoch = args.num_samples_per_epoch
    sampling_epoch = args.num_sampling_epoch
    im_dir = args.im_dir

    print("model loading done")
    test_score = []
    gen_score = []
    gen_score_alldata = []
    gen_score_allreq = []
    test_mismatch = []
    gen_mismatch = []
    gen_mismatch_alldata = []
    gen_mismatch_allreq = []
    results = {}
    for r in req:
        print(f'Starting calculation for requirement {r}')
        results[r] = {}
        if dataset =='mnist':
            ori_dataloader, _, model1, model2, feature1, feature2 = mnist(r)
        elif dataset == 'celeba':
            ori_dataloader, _, model1, model2, feature1, feature2 = celeba(r)
        print(f"Calculating success rate over testing dataset for {r}.")
        ori_score, mismatch_index, ori_img_index, ori_pred1, ori_pred2, ori_labels1, ori_labels2 = compute_match(dataloader = ori_dataloader, model1 = model1, 
                                model2 = model2, device = device, feature1 = feature1, feature2 = feature2)
        test_score.append(ori_score*100)
        test_mismatch.append(mismatch_index)

        score = []
        score_alldata = []
        score_allreq = []
        r_mismatch = []
        r_mismatch_alldata = []
        r_mismatch_allreq = []
        print(f"Calculating success rate over generated images for {r}.")
        img_in = []
        img_in_alldata = []
        img_in_allreq = []
        pred1 = []
        pred1_alldata = []
        pred1_allreq = []
        pred2 = []
        pred2_alldata = []
        pred2_allreq = []
        for i in range(sampling_epoch):
            print(f'generating image for turn {i+1} for {r}')
            print("score calculation")
            if dataset =='mnist':
                _, gen_dataloader, _, _, _, _ = mnist(r, flag = False, turn = i)
                _, gen_dataloader_alldata, _, _, _, _ = mnist(r, flag = False, turn = i, datatype = "alldata")
                _, gen_dataloader_allreq, _, _, _, _ = mnist(r, flag = False, turn = i, datatype = "allreq")
                generated_score_alldata, mismatch_index_alldata, gen_img_index_alldata, gen_pred1_alldata, gen_pred2_alldata, _, _ = compute_match(dataloader = gen_dataloader_alldata, model1 = model1, 
                            model2 = model2, device = device)
                generated_score_allreq, mismatch_index_allreq, gen_img_index_allreq, gen_pred1_allreq, gen_pred2_allreq, _, _ = compute_match(dataloader = gen_dataloader_allreq, model1 = model1, 
                            model2 = model2, device = device)
            
            elif dataset == 'celeba':
                _, gen_dataloader, _, _, _, _ = celeba(r, flag = False, turn = i)
            generated_score, mismatch_index, gen_img_index, gen_pred1, gen_pred2, _, _ = compute_match(dataloader = gen_dataloader, model1 = model1, 
                            model2 = model2, device = device)
   
            score.append(generated_score*100)
            img_in.extend(gen_img_index)
            pred1.extend(gen_pred1)
            pred2.extend(gen_pred2)
            if dataset =='mnist':
                score_alldata.append(generated_score_alldata*100)
                img_in_alldata.extend(gen_img_index_alldata)
                pred1_alldata.extend(gen_pred1_alldata)
                pred2_alldata.extend(gen_pred2_alldata)

                score_allreq.append(generated_score_allreq*100)
                img_in_allreq.extend(gen_img_index_allreq)
                pred1_allreq.extend(gen_pred1_allreq)
                pred2_allreq.extend(gen_pred2_allreq)
        gen_score.append(score)
        if dataset == "mnist":
            gen_score_alldata.append(score_alldata)
            gen_score_allreq.append(score_allreq)
            results[r][feature1] = {
                    'test_pred': ori_pred1,
                    'test_image_paths': ori_img_index,
                    'test_labels': ori_labels1,
                    'gen_pred': pred1,
                    'gen_image_paths': img_in,
                    'gen_pred_alldata': pred1_alldata,
                    'gen_image_paths_alldata': img_in_alldata,
                    'gen_pred_allreq': pred1_allreq,
                    'gen_image_paths_allreq': img_in_allreq
                }
            results[r][feature2] = {
                    'test_pred': ori_pred2,
                    'test_labels': ori_labels2,
                    'gen_pred': pred2,
                    'gen_pred_alldata': pred2_alldata,
                    'gen_pred_allreq': pred2_allreq
                }
        else:
            results[r][feature1] = {
                    'test_pred': ori_pred1,
                    'test_image_paths': ori_img_index,
                    'test_labels': ori_labels1,
                    'gen_pred': pred1,
                    'gen_image_paths': img_in
                }
            results[r][feature2] = {
                    'test_pred': ori_pred2,
                    'test_labels': ori_labels2,
                    'gen_pred': pred2
                }

    print(f'Success rate over test dataset: {test_score}')
    print(f'Success rate over generated images:')
    print(gen_score)
    scores = {}
    scores['test'] = test_score
    scores['gen'] = gen_score
    if dataset == "mnist":
        scores['gen_fulldata'] = gen_score_alldata
        score['gen_fullreq'] = gen_score_allreq

    print(f'\n\nmismatch from test set: {test_mismatch}')

    with open(f"results/rq1_{args.dataset}_fulldata.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(f"results/rq1_{args.dataset}.json", "w") as f:
        json.dump(scores, f, indent=4)
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq1')
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--im_dir', default= 'sample_gen', type = str)
    parser.add_argument('--num_samples', default = 100, type = int)
    parser.add_argument('--num_samples_per_epoch', default = 20, type = int)
    parser.add_argument('--num_sampling_epoch', default = 10, type = int)
    args = parser.parse_args()

    main(args)

