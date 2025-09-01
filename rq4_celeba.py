import os
from PIL import Image 
import torch
from torchvision import transforms
import sys
from train_classifier import MyVitModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_location = "output/celeba_binary_classifier_LC/eyeglasses.pth"
model = MyVitModel().to(device)
model.load_state_dict(torch.load(model_location))
model.eval()

im_size = 384
##Transforms function to transform the input images for this model
transform = transforms.Compose([
                            transforms.Resize(im_size, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                        ])
                        
def rq4(img_dir, file_accuracy, file_fault):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    assert len(img_files) == 10000, "incorrect sample size"
    if os.path.exists(file_accuracy):
        os.remove(file_accuracy)
    if os.path.exists(file_fault):
        os.remove(file_fault)
    for it in range(10):
        pass_cases = []
        fail_cases = []
        with torch.no_grad():
            images = [Image.open(os.path.join(img_dir, f"{fname}.png")) for fname in range(it*1000, (it+1)*1000)]
            for index, img in enumerate(images):
                img_tensor = transform(img).to(device)
                batch = img_tensor.unsqueeze(0)
                logits, probas = model(batch)
                _, predicted_labels = torch.max(probas, 1)
                p = (predicted_labels == 1)
                if p.int().item() == 1:
                    pass_cases.append(it*1000+index)
                else:
                    fail_cases.append(it*1000+index)
                
        with open(file_accuracy, "a") as f:
            f.write(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000)+"\n")
        with open(file_fault, "a") as f:
            f.write(f"Iteration {it} fail cases : "+str(fail_cases)+"\n")
        print(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000))
    
    
for req in range(1, 8):
    print(f"Requirement {req} ...")
    ###For LoRA
    file_accuracy = f"results/celeba_r{req}_passrate.txt"
    file_fault = f"results/celeba_r{req}_failcases.txt"    
    img_dir = f"output/flux_lora_celeba_r{req}/sample_gen"
    
    rq4(img_dir, file_accuracy, file_fault)
    ###For VQA LoRA
    file_accuracy = f"results/celeba_r{req}_minicpm_passrate.txt"
    file_fault = f"results/celeba_r{req}_minicpm_failcases.txt"    
    img_dir = f"output/flux_lora_celeba_minicpm_r{req}/sample_gen"
    
    rq4(img_dir, file_accuracy, file_fault)
    

    
    
    