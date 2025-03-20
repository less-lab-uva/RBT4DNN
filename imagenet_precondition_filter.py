import pandas as pd 
from tqdm import tqdm
from PIL import Image
from glob import glob
import os

'''
isMammal:	        hasFurOrHair and not hasFeathers and not hasWings and not has6Legs					
isBovid:	        isMammal and hasHorns and hasHooves and has4Legs and not inWater					
isAquaticMammal:    isMammal and inWater					
isBird:	            hasFeathers and not hasFurOrHair and hasBeak and hasWings and not has4Legs					
isAquaticBird:	    isBird and inWater					
isInsect:	        hasAntenna and has6Legs and not has8Legs and not hasFeathers and not hasFurOrHair					
isArachnid:	        has8Legs and not hasAntenna and not hasWings and not hasFeathers and not hasFurOrHair	

Precondition	        Postcondition
isBovid	                BovidClasses
isBird	                BirdClasses
isAquaticBird	        AquaticBirdClasses
isInsect	            InsectClasses
isArachnid	            ArachnidClasses
isAquaticMammal	        AquaticMammalClasses
isInsect and hasWings	WingedInsect

columns = ["inWater", "hasWings", "has2Legs", "has4Legs", "has6Legs", "has8Legs", 
                "hasFurOrHair", "hasFeathers", "hasHorns", "hasHooves", "hasBeak", "hasAntenna", "isAnimal", "hasHuman"]
'''
def noLeg(df):
    result = df.loc[(df["has2Legs"]==0) & (df["has4Legs"]==0) & (df["has6Legs"]==0) & (df["has8Legs"]==0)]
    return result
def isMammal(df):
    result = df.loc[(df["hasFurOrHair"]==1) & (df["hasFeathers"]==0) & (df["hasWings"]==0) & (df["has6Legs"]==0) & (df["has8Legs"]==0)]
    return result
def isBovid(df):
    df1 = isMammal(df)
    result = df1.loc[(df1["hasHooves"]==1) & (df1["has4Legs"] == 1) & (df1["hasTrueHorns"] == 1) 
                    & (df1["inWater"] == 0) & (df1["hasHuman"]==0)]
    return result

def isUngulate(df):
    df1 = isMammal(df)
    result = df1.loc[(df1["hasHooves"]==1) & (df1["hasRigidHooves"]==1) & (df1["has4Legs"] == 1) & (df1["hasEqualLegs"] == 1)
                    & (df["has6Legs"]==0) & (df["has8Legs"]==0) & (df1["inWater"] == 0) & (df1["hasHuman"]==0) 
                    & (df1["hasScales"]==0)  & (df1["hasBeak"]==0) & (df1["hasAntenna"]==0)]
    return result
def isAquaticMammal(df):
    df1 = isMammal(df)
    result = df1.loc[(df1["inWater"]==1) & (df1["hasHuman"]==0)]
    return result
def isBird(df):
    result = df.loc[(df["hasFeathers"]==1) & (df["hasFurOrHair"]==0) & (df["hasBeak"]==1) 
                    & (df["hasWings"]==1) & (df["has2Legs"]==1) & (df["has4Legs"]==0) 
                    & (df["has6Legs"]==0) & (df["has8Legs"]==0) & (df["hasHuman"]==0)]
    return result
def isAquaticBird(df):
    df1 = isBird(df)
    result = df1.loc[df1["inWater"]==1]
    return result

def isInsect(df):
    mam = isMammal(df)
    result = df.loc[(df["hasExoskeleton"]==1) & (df["hasAntenna"]==1) & (df["has3SegmentedBody"]==1) 
            & (df["has6Legs"]==1) & (df["has2Legs"]==0) & (df["has4Legs"]==0) 
            & (df["has8Legs"]==0) & (df["hasFeathers"]==0) & (df["hasHuman"]==0)]
    return result.drop(mam.index, errors = 'ignore')
def isArthropod(df):
    mam = isMammal(df)
    result = df.loc[(df["hasExoskeleton"]==1) & (df["hasFeathers"]==0) & (df["hasHuman"]==0) & (df["hasScales"] == 0)
                    & (df["hasFins"]==0)]
    return result.drop(mam.index, errors = 'ignore')

def isArachnid(df):
    result = df.loc[(df["has8Legs"] == 1) & (df["hasAntenna"] == 0) & (df["hasWings"] == 0) & (df["hasFeathers"] == 0) & (df["hasFurOrHair"] == 0)]
    return result

def isWingedInsect(df):
    df1 = isInsect(df)
    result = df1.loc[(df1["hasWings"]==1)]
    return result

def isSnake(df):
    result = df.loc[(df["hasScales"] == 1) & (df["hasSmoothScales"] == 1) & (df["hasEyes"] == 1) & (df["hasLeg"] == 0) & (df["has6Legs"]==0) & (df["has8Legs"]==0)
                    & (df["has2Legs"]==0) & (df["has4Legs"]==0) & (df["hasElongatedCylindricalBody"]==1) 
                    & (df["hasHuman"]==0) & (df["hasFins"]==0) & (df["hasExoskeleton"]==0) & (df["hasWings"] == 0) 
                    & (df["hasAntenna"] == 0) & (df["hasFeathers"] == 0) & (df["hasFurOrHair"] == 0) 
                    & (df["hasEqualLegs"] == 0) & (df["hasHooves"] == 0) & (df["hasRigidHooves"] == 0)]
    return result



def main():
    file_path = "rbt4dnn/imagenet/train/SingleRealAnimal.csv"
    folder_path = "rbt4dnn/imagenet"
    col = 'isSnake'
    text_list ={
        "isBovid" : "a bovid",
        "isSnake" : "a snake",
        "isUngulate" : "an ungulate",
        "isAquaticMammal" : "an aquatic mammal",
        "isBird" : "a bird",
        "isAquaticBird" : "an aquatic bird",
        "isInsect" : "an insect",
        "isArachnid" : "an arachnid",
        "isWingedInsect" : "an insect with wings",
        "isArthropod" : "an arthropod"
    }

    df = pd.read_csv(file_path)
    df.set_index('index', inplace = True)
    
    
    result = globals()[col](df)
    print('saving results!!!!')
    result.to_csv(os.path.join(folder_path, f"{col}.csv"), index_label = 'index')
    
    df = pd.read_csv(os.path.join(folder_path,f"{col}.csv"))
    df.set_index('index', inplace = True)
    des = os.path.join(folder_path,f"{col}")
    if os.path.exists(des):
        fnames = glob(os.path.join(des, "*.JPEG"))
        print("Total images in the folder: ", len(fnames))
        print("deleting the folder!!!")
        for filename in os.listdir(des):
            file_path = os.path.join(des, filename)
            
            # Check if it is a file (not a subdirectory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
        os.rmdir(des)
    os.mkdir(des)
    text = f"Imagenet. The image has {text_list[col]}."

    resize_size = 224  # ImageNet standard
    for i in tqdm(range(len(df.index))):
        fol = df.index[i].split('_')[0]
        img_name = df.index[i].split('.')[0]
        image = Image.open(os.path.join(folder_path,f"train",fol,df.index[i]))

        # Resize while maintaining aspect ratio    
        image = image.resize((resize_size, resize_size), Image.Resampling.BICUBIC)

        image.save(os.path.join(des,df.index[i]))
        image.close()
        with open(os.path.join(des, f"{img_name}.txt"),"w") as f:
            f.write(text)


if __name__ == "__main__":
    main()