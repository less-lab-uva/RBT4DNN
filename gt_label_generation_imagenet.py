import glob
from tqdm import tqdm
import pandas as pd
import os
import torch
import argparse
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def load_model():
    # load omni model default, the default init_vision/init_audio/init_tts is True
    # if load vision-only model, please set init_audio=False and init_tts=False
    # if load audio-only model, please set init_vision=False
    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-o-2_6',
        trust_remote_code=True,
        attn_implementation='sdpa', # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False
    )


    
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

    # In addition to vision-only mode, tts processor and vocos also needs to be initialized
    # model.init_tts()
    return model, tokenizer
def make_base_csv(datapath, columns):
    folders = os.listdir(datapath)
    fnames = []
    print("collecting images")
    for f in folders:
        fnames += glob.glob(os.path.join(datapath, f, f'*.JPEG'))
    print("total files: ",len(fnames))
    files = []
    for f in fnames:
        files.append(f.split("/")[-1])
    # print("filenames: ", files)
    print("total data: ",len(files))
    df = pd.DataFrame(columns = columns, index = files)
    val = [0 for i in range(len(columns))]
    print("initiating the dataframe with zeros")
    for i in df.index:
        df.loc[i] = val
    df.to_csv(os.path.join(datapath, "metadata.csv"), index_label = 'index')

def load_column_values(datapath, base_df_name, question, col_name, start, end):
    df = pd.read_csv(os.path.join(datapath, base_df_name))
    df.set_index('index', inplace = True)
    col = []
    col.append(col_name)
    col_df = pd.DataFrame(index = df.index, columns = col)
    index_list = list(col_df.index)
    
    ######load model##########
    model, tokenizer = load_model()
    model = model.eval().cuda()

    ######figure out data######
    for i in tqdm(range(start, end)):
        folder = index_list[i].split("_")[0]
        img_loc = os.path.join(datapath, folder,index_list[i])
        image = Image.open(img_loc).convert("RGB")
        msgs = [{'role': 'user', 'content': [image, question]}]
        answer = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
        )
        if answer.lower().startswith("yes"):
            col_df.loc[index_list[i]] = 1
        else:
            col_df.loc[index_list[i]] = 0
    col_df.to_csv(os.path.join(datapath, f"{col_name}_{start}_{end}.csv"), index_label = 'index')
    
def main(args):
    datapath = "rbt4dnn/imagenet/train/"
    question = {
        "hasExoskeleton" : "Does the object have exoskeleton? Answer only yes or no.",
        "hasElongatedCylindricalBody" : "Does the object have elongated and cylindrical body shape? Answer only yes or no.",
        "hasEqualLegs" : "Are the object's front and hind legs equal in length? Answer only yes or no.",
        "inWater": "Is the object in water? Answer only yes or no.",
        "LiveinWater" : "Does the object primarily live in water? Answer only yes or no.",
        "hasWings" : "Does the object have wings? Answer only yes or no.",
        "hasLeg" :  "Does the object have any leg? Answer only yes or no.",
        "has2Legs" : "Does the object have 2 legs? Answer only yes or no.",
        "has4Legs" : "Does the object have 4 legs? Answer only yes or no.",
        "has6Legs" :  "Does the object have 6 legs? Answer only yes or no.",
        "has8Legs" : "Does the object have 8 legs? Answer only yes or no.", 
        "hasFurOrHair" : "Does the object have fur or hair? Answer only yes or no.",
        "hasFeathers" : "Does the object have feathers? Answer only yes or no.",
        "hasHorns" :  "Does the object have horns? Answer only yes or no.",
        "hasTrueHorns" : "Does the object usually have true horns? Answer only yes or no.",
        "hasBovineHorns":  "Does the object usually have bovine horns? Answer only yes or no.",
        "hasHooves" :  "Does the object have a hooves? Answer only yes or no.",
        "hasRigidHooves": "Does the object have rigid hooves? Answer only yes or no.",
        "hasBeak" : "Does the object have a beak? Answer only yes or no.",
        "hasAntenna" : "Does the object have antennae? Answer only yes or no.", 
        "isAnimal" : "Is the object an animal? Answer only yes or no.",
        "hasTailWithTuft" : "Does the object usually have a tail with a tuft? Answer only yes or no.",
        "hasSnoutWithBluntEnd": "Does the object have a snout with a blunt end? Answer only yes or no.",
        "hasMammaryGlands" : "Does the object that is not a human have a mammary glands? Answer only yes or no.",
        "isSingleRealAnimal" : "Does the image have only one real, living animal? Answer only yes or no.",
        "hasHuman" : "Does the image have any human being? Answer only yes or no.",
        "isUnderWater": "Does the object is under the water? Answer only yes or no.",
        "hasScales" : "Is the object cover in scales? Answer only yes or no.",
        "hasSmoothScales" : "Is the object cover in smooth scales? Answer only yes or no.",
        "has3SegmentedBody": "Does the animal's body have distinct segments only for head, thorax and abdomen and no other segments? Answer only yes or no.",
        "hasFins" : "Does the object have fins? Answer only yes or no.",
        "hasEyes" : "Does the object have eyes? Answer only yes or no."
    }
    # ranges = [0, 320291, 640582, 960873, 1281167]
    ranges = [0, 119510, 239020, 358530, 478043 ]
    ########label to run: hasBovineHorns
    #######Create the base dataframe
    # make_base_csv(datapath, columns)
    # base_df_name = 'metadata.csv'
    base_df_name = "SingleRealAnimal.csv"
    col_name = args.column

            
    load_column_values(datapath, base_df_name, question[col_name], col_name, ranges[args.start], ranges[args.end])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Argument for Imagenet metadata creation")
    parser.add_argument('--column', default = "hasBeak", type = str)
    parser.add_argument('--start', default = 0, type = int)
    parser.add_argument('--end', default = 4, type = int)
    args = parser.parse_args()
    main(args)

