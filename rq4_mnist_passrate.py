from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from PIL import Image 
import torch

#Cleanup old logs
for req in range(1,7):
    if os.path.exists(f"results/mnist_r{req}_passrate.txt"):
        os.remove(f"results/mnist_r{req}_passrate.txt")
    
target_labels = [2,3,7,9,6,0]
for req in range(1, 7):
    img_dir = f"rq4/output/flux_lora_mnist_r{req}/rq4_samples"
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    assert len(img_files) == 10000, "incorrect sample size"

    print(f"Requirement {req} ...")
    exp_label = target_labels[req-1]

    processor = AutoImageProcessor.from_pretrained("farleyknight-org-username/vit-base-mnist")
    model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
    
    for it in range(10):
        images = [Image.open(os.path.join(img_dir, f"{fname}.png")) for fname in range(it*1000, (it+1)*1000)]
        inputs = processor(images, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = [l.argmax(-1).item() for l in logits]
        fail_cases = [(it*1000+i, exp_label, p) for i, p in enumerate(predicted_label) if p != exp_label]
                
        with open(f"results/mnist_r{req}_passrate.txt", "a") as f:
            f.write(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000)+"\n")
        print(f"Iteration {it} Pass Rate : "+str((1000-len(fail_cases))/1000))
