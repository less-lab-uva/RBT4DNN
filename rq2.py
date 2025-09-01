import os
import torch
import torch_fidelity
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from rq1 import find_r_data
from torchvision import transforms

from torchmetrics.image.kid import KernelInceptionDistance

def main(args):
    req = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7']
    for r in req:
        print(f'calculating KID for {r}\n\n')

        input1 = f'data/{args.dataset}_{r}'

        input2 = f'output/flux_{args.dataset}_{r}/sample_gen'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=input1, 
            input2=input2, 
            cuda=torch.cuda.is_available(), 
            kid=True,
            kid_subset_size = 100
        )
        print(metrics_dict)
        print('\n\n')

    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments for rq2')
    parser.add_argument('--dataset', default='mnist', type=str)

    args = parser.parse_args()
    
    main(args)