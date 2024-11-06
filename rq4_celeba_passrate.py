import os
from PIL import Image 
import torch
from torchvision import transforms
import sys
from rq1.train_classifier import MyVitModel

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
                        
#Cleanup old logs
for req in range(1,7):
    if os.path.exists(f"celeba_r{req}_faults.txt"):
        os.remove(f"celeba_r{req}_faults.txt")
    if os.path.exists(f"celeba_r{req}_passrate.txt"):
        os.remove(f"celeba_r{req}_passrate.txt")
    
for req in range(1, 7):
    img_dir = f"rq4/output/flux_lora_celeba_r{req}/rq4_samples"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    assert len(img_files) == 10000, "incorrect sample size"

    print(f"Requirement {req} ...")
    
    for it in range(10):
        pass_cases = []
        fail_cases = []
        pred = []
        with torch.no_grad():
            images = [Image.open(os.path.join(img_dir, f"{fname}.png")) for fname in range(it*1000, (it+1)*1000)]
            for index, img in enumerate(images):
                img_tensor = transform(img).to(device)
                batch = img_tensor.unsqueeze(0)
                logits, probas = model(batch)
                _, predicted_labels = torch.max(probas, 1)
                p = (predicted_labels == 1)
                pred.append(p.int().item())
                if p.int().item() == 1:
                    pass_cases.append(it*1000+index)
                else:
                    fail_cases.append(it*1000+index)
                
        if len(fail_cases) > 0:
            with open(f"celeba_r{req}_faults.txt", "a") as f:
                result = f"Iteration {it} Faults : "
                for fc in fail_cases:
                    result = result + str(fc) + ","
                f.write(result+"\n")
                print(result)
                
        with open(f"celeba_r{req}_passrate.txt", "a") as f:
            f.write(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000)+"\n")
        print(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000))
        
