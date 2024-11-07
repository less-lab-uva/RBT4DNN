# RBT4DNN
This repository contains the code and data for the framework "RBT4DNN".

## Requirements
![requirements](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/requirements.png)

## Result Data
The results of our experiments can be found in [results](https://github.com/nusratdeeptee/RBT4DNN/tree/main/results). The description of the files are as follows:

- rq1_<em>[dataset]</em>.json: Contains the test and generated data percentage match for the dataset. The dataset value can be mnist, celeba or sgsm.
- rq1_<em>[dataset]</em>_fulldata.json: Contains the detail results for the dataset. For each requirement and for each image, this file contains the glossary term classifiers' decision with the image id.
-  <em>[dataset]</em>_rq2.txt: Contains the KID score for each requirements of a dataset.
-  <em>[dataset]</em>_rq3.txt: contains the detail JS divergence calculation for each requirements of a dataset.
-  <em>[dataset]</em>_r<em>[req]</em>_passrate.txt: Contains passrates for 10 repetitions of the RQ4 study. req is 1-6 for the MNIST and CelebA-HQ datasets and 1-7 for the SGSM dataset. 

### RQ1:
![rq1](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq1.png)

### RQ2:
![rq2](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq2.png)

### RQ3:
![rq3](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq3.png)

## Usage

To reproduce the results, First create a python virtual environment (our used version is python 3.11.6). Then, activate the environment and inside the environment, run the following command:

<code>pip install -r requirements.txt</code>

### Dataset Preparation

For MNIST, download the train images and train-morpho.csv from here [MorphoMNIST](https://github.com/dccastro/Morpho-MNIST). Put the images in 'data/MNIST/train_images'

For CelebA-HQ, download the images and the 40 attribute annotations list from here [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

For SGSM, download the data from here [SGSM](https://github.com/less-lab-uva/SGSM)

To have train and test sets, run the following command:

<code>python make_train_test_dataset.py</code>

Note that, the sets are randomly generated, hence it is possible to observe little deviation in the results from our reported results. We reported the train and test sets used in our experiments in [data](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data) so that one can reproduce the exact results.

### Train LoRA

To train a LoRA, we first need to create a folder with images and their associated text for each requirement. Use the following codes to create the folder for MNIST and CelebA-HQ.

<code>python create_mnist_image_datafolder.py</code>

<code>python create_celeba_image_datafolder.py</code>

For SGSM, run the ipynb file: <code>create_sgsm_image_datafolder.ipynb</code>

To train a LoRA, follow the steps from here [LoRA](https://github.com/ostris/ai-toolkit/)

### Generate Images from Lora

We provided 100 generated images from the per-requirement LoRA for each requirement of each dataset in [images_from_loras](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras). To produce more images, use the following code.

<code>python sample_from_lora.py --dataset [dataset] --num_samples [num_of_samples_to_generate] --num_samples_per_epoch [num_of_samples_to_generate_at_once] --req [list_of_requirements]</code>


### Train Glossary Term Classifier

Use the following code to train the glossary term classifier for a glossary term.

<code>python train_classifier.py --fet [glossary_term] --dataset [dataset]</code>

The pretrained models used in the paper can be found here [trained models](https://zenodo.org/records/14047510)

### Run RQ Studies

To Produce the RQ results, run the following commands

For RQ1:

<code> python rq1.py</code> (for MNIST and CelebA-HQ)

run <code>rq1_sgsm.ipynb</code> for SGSM

For RQ2:

<code> python rq2.py</code>

For RQ3:

<code>python rq3.py</code>

For RQ4:

Before running SGSM experiment, download ComfyUI code from (https://github.com/comfyanonymous/ComfyUI) to the project directory.
Copy pretrained SGSM loras from [trained models](https://zenodo.org/records/14047510) to ComfyUI/models/loras/ directory.

Follow the instructions in (https://comfyanonymous.github.io/ComfyUI_examples/flux/) and copy the flux models to the appropriate ComfyUI subdirectories.

Copy driving_model.ckpt from [trained models](https://zenodo.org/records/14047510) to <em>output</em> directory.

<code>python rq4_gen_samples.py --req r<em>[1,6]</em> --dataset <em>[mnist,celeba]</em></code> 

<code>python rq4_sgsm.py --req r<em>[1,7]</em></code> 
  
<code>python rq4_mnist_passrate.py</code>

<code>python rq4_celeba_passrate.py</code>


