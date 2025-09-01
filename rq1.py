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
from sample_from_lora import sample_images
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def find_r_data(test_df, feature1, feature2 = None):
    ##############compute pos data for a feature the index of the dataframe is image name##########
    pos = []
    for i in range(len(test_df.index)):
        if isinstance(feature1, list):
            flag = 0
            for f in feature1:
                if test_df[f].iloc[i] == 1:
                    flag = 1
                    break
            if flag == 1:
                if feature2 is None or test_df[feature2].iloc[i] == 1:
                    pos.append(test_df.index[i])

        else:
            if feature2 is not None and test_df[feature1].iloc[i] == 1 and test_df[feature2].iloc[i] == 1:
                pos.append(test_df.index[i])
            elif feature2 is None and test_df[feature1].iloc[i] == 1:
                pos.append(test_df.index[i])

    print("\ntotal number of data: ", len(test_df.index))
    print('total number of pos data for ', feature1," and ", feature2, ": ", len(pos))

    return pos

def compute_match(dataloader, model1, model2, device, feature1=None, feature2=None):
    sample_size = 0
    correct = 0
    at_least_one_positive_match = 0
    at_least_one_prediction_positive = 0
    mismatch = []

    img_list = []
    pred1 = []   # Will store list of predictions for each model in model1
    pred2 = []
    labels1 = []  # Will store list of labels for each feature in feature1
    labels2 = []

    is_multi_model1 = isinstance(model1, list)

    with torch.no_grad():
        for images, data in dataloader:
            images = images.to(device)
            labels, img_index = data
            img_list.extend(img_index)

            if is_multi_model1:
                label1_list = []
                for f in range(len(model1)):
                    if feature1 is not None:
                        label = labels[feature1[f]]
                    else:
                        label = torch.ones(images.shape[0])
                    label1_list.append(label.to(device))
                for i in range(images.shape[0]):
                    labels1.append([label1_list[j][i].item() for j in range(len(model1))])
            else:
                if feature1 is not None:
                    label1 = labels[feature1]
                else:
                    label1 = torch.ones(images.shape[0])
                label1 = label1.to(device)
                labels1.extend(label1.tolist())

            if feature2 is not None:
                label2 = labels[feature2]
            else:
                label2 = torch.ones(images.shape[0])
            label2 = label2.to(device)
            labels2.extend(label2.tolist())

            if is_multi_model1:
                preds1 = []
                for m in model1:
                    _, probas = m(images)
                    _, pred = torch.max(probas, 1)
                    preds1.append(pred)
                    pred1.append(pred.tolist())
            else:
                _, probas1 = model1(images)
                _, predicted_labels1 = torch.max(probas1, 1)
                pred1.extend(predicted_labels1.tolist())

            if model2 is not None:
                _, probas2 = model2(images)
                _, predicted_labels2 = torch.max(probas2, 1)
                pred2.extend(predicted_labels2.tolist())

            for i in range(images.shape[0]):
                sample_size += 1
                is_exact_match = True
                has_positive_match = False
                has_positive_prediction = False

                if is_multi_model1:
                    for j in range(len(model1)):
                        pred_val = preds1[j][i].item()
                        label_val = label1_list[j][i].item()

                        if pred_val != label_val:
                            is_exact_match = False
                        if pred_val == 1:
                            has_positive_prediction = True
                        if pred_val == 1 and label_val == 1:
                            has_positive_match = True
                else:
                    pred_val = predicted_labels1[i].item()
                    label_val = label1[i].item()
                    is_exact_match = (pred_val == label_val)
                    has_positive_prediction = (pred_val == 1)
                    has_positive_match = (pred_val == 1 and label_val == 1)

                if model2 is not None:
                    if predicted_labels2[i] != label2[i]:
                        is_exact_match = False
                        has_positive_match = False
                        has_positive_prediction = False

                if is_exact_match:
                    correct += 1
                else:
                    if is_multi_model1 and feature1 is None and has_positive_prediction:
                        correct += 1
                    else:
                        mismatch.append(img_index[i])
                if has_positive_match:
                    at_least_one_positive_match += 1
                if has_positive_prediction:
                    at_least_one_prediction_positive += 1

    print(f'Total samples: {sample_size}')
    print(f'All-feature exact matches: {correct}')
    print(f'At least one feature with positive match: {at_least_one_positive_match}')
    print(f'At least one feature with prediction = 1: {at_least_one_prediction_positive}')

    return (
        correct / sample_size,
        mismatch,
        img_list,
        pred1,
        pred2,
        labels1,
        labels2,
        at_least_one_positive_match / sample_size,
        at_least_one_prediction_positive / sample_size,
    )




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mnist(req, flag = True, turn = None, isSingleLora= False, isReqDataOnly = False):
    im_size = 64
    batch_size = 64
    transform = transforms.Compose([
                        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    if flag: ##if running the test dataset
        '''
        test_label_path = 'data/MNIST/test.csv'
        test_df = pd.read_csv(test_label_path)
        test_df = test_df.set_index('images')
        im_path = 'data/MNIST/train_images_ori'
        '''
        test_label_path = 'data/MNIST/original_test_data_for_requirements.csv'
        test_df = pd.read_csv(test_label_path)
        test_df = test_df.set_index('images')
        im_path = 'data/MNIST/test_images'
        # print("im_names: ")
        # print(im_names)
        feature_name = {
            'r1': 'Very_Low_Height',
            'r2': 'Very_Thick',
            'r3': 'Very_Thick',
            'r4': 'Very_Left_Leaning',
            'r5': 'Very_Right_Leaning',
            'r6': 'Very_Low_Height',
            'r7': ['Very_Thin', 'Very_Thick'],
        }
        digit = {
        'r1': '2',
        'r2': '3',
        'r3': '7',
        'r4': '9',
        'r5': '6',
        'r6': '0',
        'r7': '8',
        }
        feature1 = feature_name[req]
        feature2 = digit[req]
        im_names = find_r_data(test_df, feature1, feature2)
        print("length of test dataset: ", len(im_names))
        ############################################################################################################
        ori_dataset = CustomDataset(im_names = im_names, label = test_df, transform = transform, 
                    im_path = im_path, feature = None, im_channel = 1)

        ori_dataloader = DataLoader(ori_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
        gen_dataloader = None
        # gen_dataset = None

        if isinstance(feature_name[req], list): ###If multiple features presented with "or" 
            model1 = []
            for i in feature_name[req]:
                model1_location = f'output/mnist_binary_classifier_{req}/{i.lower()}.pth'
                md = ResnetModel().to(device)
                md.load_state_dict(torch.load(model1_location, weights_only = True))
                md.eval()
                model1.append(md)
        else: ##for single feature
            model1_location = f'output/mnist_binary_classifier_{req}/{feature_name[req].lower()}.pth'
            model1 = ResnetModel().to(device)
            model1.load_state_dict(torch.load(model1_location, weights_only = True))
            model1.eval()
        #########################################################################################################################

        model2_location = f'output/mnist_binary_classifier_{req}/{digit[req]}.pth'

        model2 = ResnetModel().to(device)
        model2.load_state_dict(torch.load(model2_location, weights_only = True))

        model2.eval()
        
    else: ####running the generated image dataset
        start = turn * 100 
        end = turn * 100 + 100 
        im_names = [f'{i}.png' for i in range(start, end)]

        if isSingleLora:
            im_path = f"output/flux_lora_mnist_all_data/{req}"
        else:
            im_path = f'output/flux_lora_mnist_{req}/sample_gen'


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
        'r6':'Wearing_Hat',
        'r7':['5_o_Clock_Shadow', 'Goatee', 'No_Beard', 'Mustache', 'Sideburns'],
    }
    
    
    model2_location = f'output/celeba_binary_classifier_1024_{req}/eyeglasses.pth'


    if flag:
        im_path = 'data/CelebA-HQ/CelebA-HQ-img'
        test_label_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-test.csv'

        test_df = pd.read_csv(test_label_path)
        test_df = test_df.set_index('images')


        feature1 = feature_name[req]
        feature2 = 'Eyeglasses'
        im_names = find_r_data(test_df, feature1, feature2)
        im_names_minicpm = find_r_data(test_df_minicpm, feature1, feature2)

        ori_dataset = CustomDataset(im_names = im_names, label = test_df, transform = transform, 
                im_path = im_path, feature = None, im_channel = 3)

        ori_dataloader = DataLoader(ori_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

        if isinstance(feature_name[req], list): ###If multiple features presented with "or" 
            model1 = []
            for i in feature_name[req]:
                model1_location = f'output/celeba_binary_classifier_1024_{req}/{i.lower()}.pth'
                md = ResnetModel().to(device)
                md.load_state_dict(torch.load(model1_location, weights_only = True))
                md.eval()
                model1.append(md)
        else: ##for single feature
            model1_location = f'output/celeba_binary_classifier_1024_{req}/{feature_name[req].lower()}.pth'
            model1 = ResnetModel().to(device)
            model1.load_state_dict(torch.load(model1_location, weights_only = True))
            model1.eval()

        

        
        model2 = ResnetModel().to(device)
        model2.load_state_dict(torch.load(model2_location, weights_only = True))

        model2.eval()
 

        

        gen_dataloader = None
        gen_dataloader_vqa = None

    else:
        start = turn * 100
        end = turn * 100 + 100
        im_names = [f'{i}.png' for i in range(start, end)]

        
        im_path_vqa = f'output/flux_lora_celeba_minicpm_{req}/sample_gen'
        
        im_path = f'output/flux_lora_celeba_{req}/sample_gen'
        gen_dataset = CustomDataset(im_names = im_names, label = None, transform = transform, 
                    im_path = im_path, feature = None, im_channel = 1)                
        
        gen_dataloader = DataLoader(gen_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        gen_dataset_vqa = CustomDataset(im_names = im_names, label = None, transform = transform, 
                    im_path = im_path_vqa, feature = None, im_channel = 1)                
        
        gen_dataloader_vqa = DataLoader(gen_dataset_vqa,
                                        batch_size=batch_size,
                                        shuffle=True)
        ori_dataloader = None
        model1 = None
        model2 = None
        feature1 = None
        feature2 = None
        
    

    return ori_dataloader, gen_dataloader, gen_dataloader_vqa,  model1, model2, feature1, feature2

def main(args):  
    print("is singleLora: ",args.isSingleLora)
    print('calculating for ', args.dataset)
    req = args.req
    dataset = args.dataset
    num_of_samples = args.num_samples 
    num_of_samples_per_epoch = args.num_samples_per_epoch
    sampling_epoch = args.num_sampling_epoch
    im_dir = args.im_dir

    test_score = []
    gen_score = []
    test_mismatch = []
    gen_mismatch = []
    if dataset == 'celeba':
        gen_score_vqa = []
    results = {}
    for r in req:
        print(f'Starting calculation for requirement {r}')
        results[r] = {}
        if dataset =='mnist':
            ori_dataloader, _, model1, model2, feature1, feature2 = mnist(r)
        elif dataset == 'celeba':
            ori_dataloader, _, _, model1, model2, feature1, feature2 = celeba(r)

        print(f"Calculating success rate over testing dataset for {r}.")
        ori_score, mismatch_index, ori_img_index, ori_pred1, ori_pred2, ori_labels1, ori_labels2, ori_at_least_one_positive_match, ori_at_least_one_prediction_positive = compute_match(dataloader = ori_dataloader, model1 = model1, 
                                model2 = model2, device = device, feature1 = feature1, feature2 = feature2)
        
        test_score.append(ori_score*100)
        test_mismatch.append(mismatch_index)
        
        score = []
        r_mismatch = []
        print(f"Calculating success rate over generated images for {r}.")
        img_in = []
        pred1 = []
        pred2 = []
        if dataset == 'celeba':
            score_vqa = []
            img_in_vqa = []
            pred1_vqa = []
            pred2_vqa = []
        for i in range(sampling_epoch):
            print(f'generating image for turn {i+1} for {r}')

            print("score calculation")
            if dataset =='mnist':
                 _, gen_dataloader, _, _, _, _ = mnist(r, flag = False, turn = i, isSingleLora = args.isSingleLora)
            elif dataset == 'celeba':
                _, gen_dataloader, gen_dataloader_vqa, _, _, _, _ = celeba(r, flag = False, turn = i)
            generated_score, mismatch_index, gen_img_index, gen_pred1, gen_pred2, _, _, _, _ = compute_match(dataloader = gen_dataloader, model1 = model1, 
                            model2 = model2, device = device)
            score.append(generated_score*100)
            img_in.extend(gen_img_index)
            pred1.extend(gen_pred1)
            pred2.extend(gen_pred2)
            if dataset == 'celeba':
                generated_score_vqa, mismatch_index_vqa, gen_img_index_vqa, gen_pred1_vqa, gen_pred2_vqa, _, _, _, _ = compute_match(dataloader = gen_dataloader_vqa, model1 = model1, 
                            model2 = model2, device = device)
                score_vqa.append(generated_score_vqa*100)
                img_in_vqa.extend(gen_img_index_vqa)
                pred1_vqa.extend(gen_pred1_vqa)
                pred2_vqa.extend(gen_pred2_vqa)
        
        gen_score.append(score)
        # Handle multiple features in feature1
        if dataset == 'celeba':
            gen_score_vqa.append(score_vqa)
            if isinstance(feature1, list):
                for i, feat in enumerate(feature1):
                    results[r][feat] = {
                        'test_pred': ori_pred1[i],
                        'test_image_paths': ori_img_index,
                        'test_labels': ori_labels1[i],
                        'test_at_least_one_positive_match': ori_at_least_one_positive_match,
                        'test_at_least_one_prediction_positive': ori_at_least_one_prediction_positive,
                        'gen_pred': pred1[i],
                        'gen_pred_vqa': pred1_vqa[i]
                    }
                results[r]['gen_image_paths'] = img_in
                results[r]['gen_image_paths_vqa'] = img_in_vqa
            else:
                results[r][feature1] = {
                    'test_pred': ori_pred1,
                    'test_image_paths': ori_img_index,
                    'test_labels': ori_labels1,
                    'gen_pred': pred1,
                    'gen_image_paths': img_in,
                    'gen_pred_vqa': pred1_vqa,
                    'gen_image_paths_vqa': img_in_vqa
                }
            results[r][feature2] = {
                    'test_pred': ori_pred2,
                    'test_labels': ori_labels2,
                    'gen_pred': pred2,
                    'gen_pred_vqa': pred2_vqa,
                }
        else:
            if isinstance(feature1, list):
                for i, feat in enumerate(feature1):
                    results[r][feat] = {
                        'test_pred': ori_pred1[i],
                        'test_image_paths': ori_img_index,
                        'test_labels': ori_labels1[i],
                        'test_at_least_one_positive_match': ori_at_least_one_positive_match,
                        'test_at_least_one_prediction_positive': ori_at_least_one_prediction_positive,
                        'gen_pred': pred1[i],
                        'gen_image_paths': img_in
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
                    'gen_pred': pred2,
                }
            

    print(f'Success rate over test dataset: {test_score}')
    
    print(f'Success rate over generated images:')
    print(gen_score)
    if dataset == 'celeba':
        print(f'Success rate over generated images for vqa data:')
        print(gen_score_vqa)
    print(f'\n\nmismatch from test set: {test_mismatch}')
    with open(f"output/rq1_{args.dataset}_new.json", "w") as f:
        json.dump(results, f, indent=4)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq1')
    parser.add_argument('--dataset', default='celeba', type=str)
    parser.add_argument('--req', nargs='+', default = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7'])
    parser.add_argument('--isSingleLora', action='store_true', help='Enable SingleLora mode')
    parser.add_argument('--no-isSingleLora', dest='isSingleLora', action='store_false', help='Disable SingleLora mode')
    parser.add_argument('--isLora', action='store_true', help='Enable Lora mode')
    parser.add_argument('--no-isLora', dest='isLora', action='store_false', help='Disable Lora mode')
    parser.add_argument('--isReqDataOnly', action='store_true', help='Enable ReqData mode')
    parser.add_argument('--no-isReqDataOnly', dest='isReqDataOnly', action='store_false', help='Disable ReqData mode')
    parser.add_argument('--im_dir', default= 'sample_gen', type = str)
    parser.add_argument('--num_samples', default = 100, type = int)
    parser.add_argument('--num_samples_per_epoch', default = 20, type = int)
    parser.add_argument('--num_sampling_epoch', default = 10, type = int)
    args = parser.parse_args()

    main(args)

