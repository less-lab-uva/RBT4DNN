from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from PIL import Image 
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
TARGET_LABEL = [2,3,7,9,6,0,8]
processor = AutoImageProcessor.from_pretrained("farleyknight-org-username/vit-base-mnist")
model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist").to(device)

def rq4(img_files, req, files_accuracy, files_faults):
    assert len(img_files) == 10000, "incorrect sample size"
    if os.path.exists(files_accuracy):
        os.remove(files_accuracy)
    if os.path.exists(files_faults):
        os.remove(files_faults)

    print(f"Requirement {req} ...")
    exp_label = TARGET_LABEL[req-1]    
    for it in range(10):
        print("iteration: ", it)
        images = [Image.open(img_files[fname]).convert("RGB") for fname in range(it*1000, (it+1)*1000)]
        print("image loading done")
        inputs = processor(images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = [l.argmax(-1).item() for l in logits]
        fail_cases = [(img_files[it*1000+i], exp_label, p) for i, p in enumerate(predicted_label) if p != exp_label]

        if len(fail_cases) > 0:
            with open(files_faults, "a") as f:
                result = f"Iteration {it} Faults : "
                for fc in fail_cases:
                    result = result + str(fc) + ","
                f.write(result+"\n")

                
        with open(files_accuracy, "a") as f:
            f.write(f"Iteration {it} Accuracy : "+str((1000-len(fail_cases))/1000)+"\n")
        print(f"Iteration {it} Accuracy : "+str((1000-len(fail_cases))/1000))

for req in range(1, 8):
    ####For Lora 
    print(f"requirement r{req}....")
    img_dir = f"output/flux_lora_mnist_r{req}/sample_gen" #For LoRA
    img_files = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('.png')]
    
    print("number of image: ", len(img_files))
    files_accuracy = f"results/mnist_r{req}_passrate.txt"
    files_faults = f"results/mnist_r{req}_faults.txt"
    rq4(img_files, req, files_accuracy, files_faults)

    img_files_deephyperion = []
    img_files_imgTrnsform = []
    #######Baselines
    for i in range(10):
        #####Deephyperion
        img_dir = f"DeepHyperion/DeepHyperion-MNIST/r{req}/run{i}/all" #For Deephyperion
        imges = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('.png')]
        img_files_deephyperion.extend(imges)
        
        ####Image transform baseline
        img_dir = f"PixelXform/MetaModelTesting/datamodels/mnist/data/images/r{req}/run{i}" #For metamodeltesting
        imges = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('.png')]
        img_files_imgTrnsform.extend(imges)
    if len(img_files_deephyperion)> 10000:
        img_files_deephyperion = img_files_deephyperion[0:10000]
    if len(img_files_imgTrnsform)> 10000:
        img_files_imgTrnsform = img_files_imgTrnsform[0:10000]
    #####Deephyperion
    files_accuracy_deephyperion = f"results/mnist_r{req}_passrate_deephyperion.txt"
    files_faults_deephyperion = f"results/mnist_r{req}_faults_deephyperion.txt"
    rq4(img_files_deephyperion, req, files_accuracy_deephyperion, files_faults_deephyperion)
    ####Image transform baseline
    files_accuracy_imageTransform = f"results/mnist_r{req}_passrate_imagetransform.txt"
    files_faults_imageTransform = f"results/mnist_r{req}_faults_imagetransform.txt"
    rq4(img_files_imgTrnsform, req, files_accuracy_imageTransform, files_faults_imageTransform)
    
        