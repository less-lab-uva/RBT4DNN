import json
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
import csv
import timm
import torch
from urllib.request import urlopen
from PIL import Image
import os
import sys
import argparse

file_loc = "rbt4dnn/imagenet/Imagenet_Metadata.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_classes():
    df = pd.read_csv(file_loc)
    classes = [c.split(",")[0] for c in list(df["Class Name"])]
    return classes

def class_id_to_label(i, labels):
    return labels[i]

def find_imagenet_leaf_nodes(synset, imagenet_synsets):
    
    leaf_nodes = []

    def dfs(node):
        hyponyms = node.hyponyms()
        if not hyponyms and node.name() in imagenet_synsets: #if no children and synset included in ImageNet
            leaf_nodes.append(node)
        else:
            for child in hyponyms:
                if child.name() in imagenet_synsets: #If synset is an internal node and included in ImageNet
                    leaf_nodes.append(child)
                else:
                    dfs(child)

    dfs(synset)
    return leaf_nodes

def find_postcondition(postcondition, sensenumber):
    imagenet_synsets = set()
    with open(file_loc, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None) #skip header
        for row in csv_reader:
            wordid = (row[0]).strip()
            pos = wordid[0]
            offset = int(wordid[1:])
            imagenet_synsets.add(wn.synset_from_pos_and_offset(pos, offset).name())
    
    internal_node = wn.synset(postcondition+".n."+sensenumber) 
    leaf_synsets = find_imagenet_leaf_nodes(internal_node, imagenet_synsets)
    postcondition_class = []
    for synset in leaf_synsets:
        postcondition_class.append(synset.name().split(".")[0].replace('_',' '))
    return postcondition_class

def load_model(modelname):
    model_type = {
        'VOLO D5' : 'volo_d5_224.sail_in1k',
        'CAFormer m36' : 'caformer_m36.sail_in1k',
        'EfficientNet B8' : 'tf_efficientnet_b8.ap_in1k'
    }
    model = timm.create_model(model_type[modelname], pretrained=True)
    model = model.eval()
    model.to(device)
    return model

def rq4(req, modelname, postclass, labels):
    if os.path.exists(f"imagenet_{req}_{modelname}_passrate.txt"):
        os.remove(f"imagenet_{req}_{modelname}_passrate.txt")
    if os.path.exists(f"imagenet_{req}_{modelname}_failcases.txt"):
        os.remove(f"imagenet_{req}_{modelname}_failcases.txt")
    if os.path.exists(f"imagenet_{req}_{modelname}_failcases.csv"):
        os.remove(f"imagenet_{req}_{modelname}_failcases.csv") 
    
    column = ["Id", "Decision_by_Model"]
    df = pd.DataFrame(columns = column)
    
    img_dir = f"output/imagenet_{req}/sample_gen"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    assert len(img_files) == 10000, "incorrect sample size"

    ###get model
    model = load_model(modelname)
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    img_per_cycle = 1000
    for it in range(10):
        pass_cases = []
        fail_cases = []
        # with torch.no_grad():
        images = [Image.open(os.path.join(img_dir, f"{fname}.png")) for fname in range(it*img_per_cycle, (it+1)*img_per_cycle)]
        for index, img in enumerate(images):
            output = model(transforms(img).to(device).unsqueeze(0))
            top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
            pred_class = labels[top5_class_indices[0].tolist()[0]]
            if pred_class.lower() in postclass:
                pass_cases.append(it*img_per_cycle+index)
            else:
                fail_cases.append(it*img_per_cycle+index)
                new_row = {column[0]: it*img_per_cycle+index, column[1]: pred_class}
                df.loc[len(df)] = new_row
                print(pred_class)
        with open(f"imagenet_{req}_{modelname}_passrate.txt", "a") as f:
            f.write(f"Iteration {it} Pass Rate : "+str((img_per_cycle-len(fail_cases))/img_per_cycle)+"\n")
        with open(f"imagenet_{req}_{modelname}_failcases.txt", "a") as f:
            f.write(f"Iteration {it} fail cases : "+str(fail_cases)+"\n")
        print(f"Iteration {it} Pass Rate : "+str((img_per_cycle-len(fail_cases))/img_per_cycle))
    df.to_csv(f"imagenet_{req}_{modelname}_failcases.csv")


def main(args):
    ######find the name of the classes
    labels = find_classes()
    #######find the postcondition classes
    postcondition = {
        "r1" : "bird",
        "r2" : "ungulate",
        "r3" : "insect",
        "r4" : "snake"
    }
    req = args.req
    modelname = ['VOLO D5', 'CAFormer m36', 'EfficientNet B8']
    postclass = find_postcondition(postcondition = postcondition[req], sensenumber = "01")

    for m in modelname:
        print(f"rq4 for model {m} starting....")
        rq4(req, m, postclass, labels)

    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "rq4 for Imagenet")
    parser.add_argument('--req', default = "r1", type = str)
    args = parser.parse_args()
    main(args)