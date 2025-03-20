
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

#######initialize pandas dataframe
columns = ["Bald", "Black_Hair", "Brown_Hair", "Eyeglasses", "Mustache", "Wavy_Hair", "Wearing_Hat"]
col_size = len(columns)
val = [0 for i in range(col_size)]
df = pd.DataFrame(columns = columns)

input_image = []
question = ["what is the color of the hair? answer shortly.", "Does the person wear eyeglasses? answer only yes or no.", 
                "is the hair wavy? answer only yes or no.", " Does the person have any big or little mustache? answer only yes or no.", 
                "Is the person bald or nearly bald? answer only yes or no.", "Is the person wearing a hat? answer only yes or no."]

data_path = 'rbt4dnn/flux/ai-toolkit/data/CelebA-HQ/' 

fnames = glob.glob(os.path.join(data_path, 'CelebA-HQ-img/*.{}'.format('jpg')))

print("fnames: ", len(fnames))
for i in range(len(fnames)):
        df.loc[i] = val
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
                if i ==0:
                        if "black" in answer.lower():
                                df.loc[indx, "Black_Hair"] = 1
                        elif "brown" in answer.lower():
                                df.loc[indx, "Brown_Hair"] = 1
                elif "yes" in answer.lower():
                        if i == 1:
                                df.loc[indx, "Eyeglasses"] = 1
                        elif i == 2:
                                df.loc[indx, "Wavy_Hair"] = 1
                        elif i == 3:
                                df.loc[indx, "Mustache"] = 1
                        elif i == 4:
                                df.loc[indx, "Bald"] = 1
                        elif i == 5:
                                df.loc[indx, "Wearing_Hat"] = 1
print("dataframe exporting to csv")
df.to_csv("rbt4dnn/flux/ai-toolkit/data/CelebA-HQ/CelebA-HQ-MiniCPM-o-2_6-labels.csv", index = True)


