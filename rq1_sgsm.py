import argparse
import torch
import json
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from train_classifier import ResnetModel
from calculate_metrics import GeneratedDataset



class TestDataset:
    def __init__(self, csv_path: Path, requirement: str):
        self.data = pd.read_csv(csv_path)
        self.features = REQUIREMENTS_FEATURES[requirement]
        self.pos = self.data[self.data[self.features].eq(1).all(axis=1)]
        # self.neg = self.data[self.data[self.features].eq(0).all(axis=1)]
        self.neg = self.data[self.data[self.features].eq(0).any(axis=1)]
        max_len = min(len(self.pos), len(self.neg))
        print(f"Requirement {requirement} has {len(self.pos)} positive samples and {len(self.neg)} negative samples. Using {max_len} samples.")
        self.images = self.pos['images'].tolist()[:max_len] + self.neg['images'].tolist()[:max_len]
        self.labels = [1]*max_len + [0]*max_len
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, self.labels[idx]

    def get_image_path(self, idx):
        return self.images[idx]

RESNET101_CLASSIFIERS = {
    'f1': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f1.pth",
    'f2': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f2.pth",
    'f3a': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f3a.pth",
    'f4a': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f4a.pth",
    'f4b': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f4b.pth",
    'f5a': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f5a.pth",
    'f6': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f6.pth",
    'f7': "/project/lesslab/rbt4dnn/sg_dataset/trained_models/resnet101/f7.pth",
}


REQUIREMENTS_FEATURES = {
    "r1": ["f1"],
    "r2": ["f2"],
    "r3": ["f3a", "f3b"],
    "r4": ["f4a", "f4b"],
    "r5": ["f5a", "f5b"],
    "r6": ["f6"],
    "r7": ["f7"]
}

REQUIREMENTS = {
    "r1": ["f1"],
    "r2": ["f2"],
    "r3": ["f3a", ("not","f1")],
    "r4": ["f4a", "f4b"],
    "r5": ["f5a", "f4b"],
    "r6": ["f6"],
    "r7": ["f7"]
}


def load_classifier(model_path, device):
    classifier = ResnetModel("resnet101")
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def get_predictions(model, loader, device, neg):
    preds = []
    with torch.no_grad():
        for img in tqdm(loader):
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).tolist()
            preds.extend(pred)
    if neg:
        preds = [1-p for p in preds]
    return preds


def get_predictions_and_labels(model, loader, device, neg):
    preds = []
    all_labels = []
    with torch.no_grad():
        for img, labels in tqdm(loader):
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).tolist()
            preds.extend(pred)
            all_labels.extend(labels.tolist())
    if neg:
        preds = [1-p for p in preds]
    return preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_set_path', type=Path, default="data/sgsm_test_dataset.csv", help='Path to the original test set split.')
    parser.add_argument('--gen_set_path', type=Path, default="data/generated_data/", help='Path to the generated data.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for r, features in REQUIREMENTS.items():
        print(f"Requirement {r}")
        results[r] = {}
        test_set = TestDataset(args.test_set_path, r)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, num_workers=4, shuffle=False)
        gen_set = GeneratedDataset(args.gen_set_path / f"{r}")
        gen_loader = torch.utils.data.DataLoader(gen_set, batch_size=256, num_workers=4, shuffle=False)
        features_names = []
        for f in features:
            neg = False
            if isinstance(f, tuple):
                neg = f[0] == "not"
                f = f[1]
            features_names.append(f)
            print(f"\tFeature {f}")
            b_classifier = load_classifier(RESNET101_CLASSIFIERS[f], device)
            test_preds, test_labels = get_predictions_and_labels(b_classifier, test_loader, device, neg)
            gen_preds = get_predictions(b_classifier, gen_loader, device, neg)
            results[r][f] = {
                'test_pred': test_preds,
                'test_labels': test_labels,
                'gen_pred': gen_preds
            }
        
        # Req labels
        results[r]["req_test_labels"] = test_set.labels

        # Calculate requirement accuracy on test set
        req_acc = []
        for i in range(len(results[r][features_names[-1]]['test_pred'])):
            if len(features_names) == 1:
                req_acc.append(results[r][features_names[0]]['test_pred'][i])
            else:
                all_ones = True
                for f_name in features_names[:-1]:
                    if not (results[r][f_name]['test_pred'][i] and results[r][features_names[-1]]['test_pred'][i]):
                        req_acc.append(0)
                        all_ones = False
                        break
                if all_ones:
                    req_acc.append(1)
        results[r]['req_test_pred'] = req_acc

        # Calculate requirement accuracy on generated set
        req_acc_gen = []
        for i in range(len(results[r][features_names[-1]]['gen_pred'])):
            if len(features_names) == 1:
                req_acc_gen.append(results[r][features_names[0]]['gen_pred'][i])
            else:
                all_ones = True
                for f_name in features_names[:-1]:
                    if not (results[r][f_name]['gen_pred'][i] and results[r][features_names[-1]]['gen_pred'][i]):
                        req_acc_gen.append(0)
                        all_ones = False
                        break
                if all_ones:
                    req_acc_gen.append(1)
        results[r]['req_acc_gen'] = req_acc_gen

    # Save results as pickle
    import pickle
    with open("rq_results/rq1.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save results as json
    with open("rq_results/rq1.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()