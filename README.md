# RBT4DNN
This repository provides the implementation and results for the framework "RBT4DNN"

## Requirements:
![requirements](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/requirements.png)

## Result Data
The results of our experiments can be found in [results](https://github.com/nusratdeeptee/RBT4DNN/tree/main/results). The description of the files are as follows:

- rq1_<em>[dataset]</em>.json: Contains the test and generated data percentage match for the dataset. The dataset value can be mnist, celeba or sgsm.
- rq1_<em>[dataset]</em>_fulldata.json: Contains the detail results for the dataset. For each requirement and for each image, this file contains the glossary term classifiers' decision with the image id.
-  <em>[dataset]</em>_rq2.txt: Contains the KID score for each requirements of a dataset.
-  <em>[dataset]</em>_rq3.txt: contains the detail JS divergence calculation for each requirements of a dataset.

### RQ1:
![rq1](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq1.png)

### RQ2:
![rq2](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq2.png)

### RQ3:
![rq3](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Figures/rq3.png)

