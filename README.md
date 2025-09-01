# RBT4DNN

RBT4DNN is a framework designed to generate high-quality test cases tailored to specific requirements, as described in the paper ["RBT4DNN: Requirements-based Testing of Neural Networks"](https://arxiv.org/abs/2504.02737). This repository showcases the usability of the proposed approach and enables the replication of the results presented in the paper.

## Overview

<p align="center">
  <img src="https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rbt4dnn_overview.png" />
</p>

## Illustration of Training and Generated Data across different requirements

|Precondition|Real Images|Generated Images|
|:----------:|:-----------:|:------------:|
|MNIST. The digit is a 7 and is very thick. |[<img src="https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/m3_ori.png" width="200"/>](m3_ori.png)| [<img src="https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/m3_gen.png" width="200"/>](m3_gen.png)|
|CelebA-HQ. The person is wearing eyeglasses and has black hair. | ![cori](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/c1_ori.png) | ![cgen](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/c1_gen.png)|
|SGSM. The ego is in the rightmost lane and is not in an intersection. | ![sori](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/s4_ori.png) | ![sgen](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/s4_gen.png)|
|Imagenet. The single animal has no limbs and no ears. | ![iori](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/isSnake_train.png) | ![igen](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/isSnake_gen.png)|

## Glossary Term Preparation

### MNIST Glossary Terms

The following table shows glossary terms for MNIST digits for different ranges of values for different Morphometric attributes with associated SNL text phrasing.

![mnist_glossary_terms](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/mnist_glossary_term.png)

We converted the Morpho-MNIST values from this [source](https://github.com/dccastro/Morpho-MNIST) into binary labels. Each SNL text entry from the table above is mapped to its corresponding binary label using the Python script [gt_label_mnist.py](https://github.com/less-lab-uva/RBT4DNN/blob/main/gt_label_mnist.py)

### CelebA-HQ Glossary Terms
The [CelebA official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) provides 40 attributes. From these, we selected a subset to define the glossary terms. To demonstrate that RBT4DNN can also work with unlabeled data, we re-labeled the dataset using the [MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) model. Label generation can be reproduced with the script [gt_label_generation_celebahq.py](https://github.com/less-lab-uva/RBT4DNN/blob/main/gt_label_generation_celebahq.py)

### SGSM Glossary Terms
For the SGSM dataset, we adopted the glossary terms from the paper: [S3C: Spatial Semantic Scene Coverage for Autonomous Vehicles](https://dl.acm.org/doi/abs/10.1145/3597503.3639178)

### ImageNet Glossary Terms
For ImageNet, we selected intermediate nodes from the [WordNet](https://wordnet.princeton.edu/) taxonomic tree—a lexical database that organizes words based on semantic relationships—that correspond to animals (e.g., bird). We then defined preconditions as combinations of morphological features, following a Zoology textbook. Specifically, we used standard morphological features of animals (e.g., feathers, wings, hooves, antennae) that distinguish levels in zoological taxa (e.g., birds, insects), and adopted these features as our glossary terms. To obtain labels for these glossary terms, we used the MiniCPM-o-2_6 model. The labeling process can be reproduced with the script [gt_label_generation_imagenet.py](https://github.com/less-lab-uva/RBT4DNN/blob/main/gt_label_generation_imagenet.py)

## Requirements (M = MNIST, C = CelebA-HQ, S = SGSM, generated images: [data/images from loras](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras))
|Id<img width=200/>|Precondition<img width=200/>|Postcondition<img width=200/>|
|:----------------:|:-------:|:-------------------------:|
|[M1](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M1)|The digit is a 2 and has very low height|label as 2|
|[M2](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M2)|The digit is a 3 and is very thick|label as 3|
|[M3](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M3)|The digit is a 7 and is very thick|label as 7|
|[M4](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M4)|The digit is a 9 and is very left leaning|label as 9|
|[M5](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M5)|The digit is a 6 and is very right leaning|label as 6|
|[M6](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M6)|The digit is a 0 and has very low height|label as 0|
|[M7](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/M7)|The digit is an 8 and is very thin or very thick|label as 8|
|[C1](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C1)|The person is wearing eyeglasses and has black hair|label as eyeglasses|
|[C2](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C2)|The person is wearing eyeglasses and has brown hair|label as eyeglasses|
|[C3](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C3)|The person is wearing eyeglasses and has a mustache|label as eyeglasses|
|[C4](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C4)|The person is wearing eyeglasses and has wavy hair|label as eyeglasses|
|[C5](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C5)|The person is wearing eyeglasses and is bald|label as eyeglasses|
|[C6](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C6)|The person is wearing eyeglasses and a hat|label as eyeglasses|
|[C7](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/C7)|The person is wearing eyeglasses and has a 5 o’clock shadow or goatee or mustache or beard or sideburns|label as eyeglasses|
|[S1](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S1)|A vehicle is within 10 meters, in front, and in the same lane|not accelerate|
|[S2](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S2)|The ego lane is controlled by a red or yellow light|decelerate|
|[S3](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S3)|The ego lane is controlled by a green light, and no vehicle is in front, in the same lane, and within 10 meters|accelerate|
|[S4](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S4)|The ego is in the rightmost lane and is not in an intersection|do not steer to the right|
|[S5](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S5)|The ego is in the leftmost lane and is not in an intersection|do not steer to the left|
|[S6](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S6)|A vehicle is in the lane to the left and within 7 meters|do not steer to the left|
|[S7](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/S7)|A vehicle is in the lane to the right and within 7 meters|do not steer to the right|
|[I1](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/I1)|The single real animal has feathers, wings, a beak, and two legs|label as a hyponym of bird|
|[I2](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/I2)|The single real animal has fur or hair, hooves, and four legs|label as a hyponym of ungulate|
|[I3](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/I3)|The single real animal has an exoskeleton, antennae, and six legs|label as a hyponym of insect|
|[I4](https://github.com/nusratdeeptee/RBT4DNN/tree/main/data/images_from_loras/I4)|The single animal has no limbs and no ears |label as a hyponym of snake|


## GTC Training

To train a Glossary Term Classifier (GTC), we first held out test data from the training data. 
To do that, we first computed the set, $D = [D_1, D_2,..., D_l]$, 
where $D_i$ is a set satisfying the requirement $i$. 
We also computed $\overline{D} = [\overline{D_1}, \overline{D_2},..., \overline{D_l}]$, where $\overline{D_i}$ is a set that does not satisfy requirement $i$. 
To construct the test set, we sorted $D$ and inserted $r$\% from the smallest set of $D$ into the test set.
Then, we moved to the next smaller set and inserted the data absent in the test set and previously considered sets. While inserting data from $D_i$, we also ensured that the amount of the data in the test set satisfying requirement $i$ is not more than $r$\% of $D_i$. 
We repeated the same procedure for $\overline{D}$ with an additional checking that $D$ did not have the inserted data. 

For each glossary term, we split the training data to include an equal number of randomly chosen inputs with and without the glossary term. 
We randomly held out 10\% of the data with and without the glossary term for the validation set. 
Then, we trained the GTC model over the filtered train set and validated it using the validation set.

## Result Data
The results of our experiments can be found in [results/](https://github.com/nusratdeeptee/RBT4DNN/tree/main/results). The description of the files are as follows:

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

### Pretrained Models

The pretrained models (the classifier and the diffusion model checkpoints) used in the paper can be found here [trained models](https://zenodo.org/records/14051679)

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

Before running SGSM experiment, download ComfyUI code from (https://github.com/comfyanonymous/ComfyUI) to the project directory. We used v0.2.6.
Copy pretrained SGSM loras from [trained models](https://zenodo.org/records/14051679) to ComfyUI/models/loras/ directory.

Follow the instructions in (https://comfyanonymous.github.io/ComfyUI_examples/flux/) and copy the flux models to the appropriate ComfyUI subdirectories.

Copy driving_model.ckpt from [trained models](https://zenodo.org/records/14051679) to <em>output</em> directory.

<code>python rq4_gen_samples.py --req r<em>[1,6]</em> --dataset <em>[mnist,celeba]</em></code> 

<code>python rq4_sgsm.py --req r<em>[1,7]</em></code> 

<code>python rq4_imagenet.py --req r<em>[1,4]</em></code> 
  
<code>python rq4_mnist_passrate.py</code>

<code>python rq4_celeba_passrate.py</code>


