
import pandas as pd
import glob
from tqdm import tqdm
import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# load omni model default, the default init_vision/init_audio/init_tts is True
# if load vision-only model, please set init_audio=False and init_tts=False
# if load audio-only model, please set init_vision=False
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa', # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True
)


model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

# In addition to vision-only mode, tts processor and vocos also needs to be initialized
model.init_tts()


columns = ["Bald", "Black_Hair", "Brown_Hair", "Eyeglasses", "Mustache", "Wavy_Hair", "Wearing_Hat", "5_o_Clock_Shadow", "Goatee", "No_Beard", "Sideburns"]

#######initialize pandas dataframe
col_size = len(columns)
data_path = 'data/CelebA-HQ/' 
csv_file = "data/CelebA-HQ/CelebA-HQ-MiniCPM-o-2_6-labels.csv"
if not os.path.exists(csv_file):
        df = pd.DataFrame(columns = columns)
else:
        df = pd.read_csv(csv_file, index_col = 'index')
for col in columns:
        df[col] = 0

input_image = []


question =["Is the person bald or nearly bald? answer only yes or no.", "What is the color of the hair? answer shortly.", 
        "Does the person wear eyeglasses? answer only yes or no.", "Does the person have any big or little mustache? answer only yes or no.",
        "is the hair wavy? answer only yes or no.", "Is the person wearing a hat? answer only yes or no."
        "Does the person have a 5 o'clock shadow? answer only yes or no.", "Does the person have a goatee? answer only yes or no.",
        "Does the person have a beard? answer only yes or no.", "Does the person have sideburns? answer only yes or no."]


fnames = glob.glob(os.path.join(data_path, 'CelebA-HQ-img/*.{}'.format('jpg')))

print("Number of images: ", len(fnames))

for i in range(len(question)):
        for fname in tqdm(fnames):
                indx = int(fname.split("/")[-1].split(".")[0])
                image = Image.open(fname).convert("RGB")
                msgs = [{'role': 'user', 'content': [image, question[i]]}]
                answer = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer
                )
                if i ==1:
                        if "black" in answer.lower():
                                df.loc[indx, "Black_Hair"] = 1
                        elif "brown" in answer.lower():
                                df.loc[indx, "Brown_Hair"] = 1
                elif "yes" in answer.lower() and colums[i + 1] != "No_Beard":
                        if i == 0:
                                df.loc[indx, "Bald"] = 1
                        else:
                                df.loc[indx, column[i+1]] = 1
                        
                elif colums[i + 1] == "No_Beard" and "yes" not in answer.lower():
                        df.loc[indx, "No_Beard"] = 1

print("dataframe exporting to csv")
df.to_csv(csv_file, index = True)
