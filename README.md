# RBT4DNN
This repository is for the update of the project "RBT4DNN"

Work progress google doc link: [RBT4DNN](https://docs.google.com/document/d/1l_r9Vw-cETf4AvpMhczbUgvMim88Z-tj9eKhTqqIEjk/edit?usp=sharing)
## Metric Score 

| Dataset | Model | cFID | Density | Coverage | Precision | Recall |
| ------ | ----- | ----- | ------ | -------- | ------- | ------- |
| CelebAHQ | Uncond dim 192 | 49.485 | 1.82 | 0.996 | 0.86 | 0.344 |
| CelebAHQ | Uncond dim 768 | 39.6 | 1.84 | 0.96 | 0.884 | 0.372 |
| MNIST | Uncond | 21.559 | 0.765 | 0.908 | 0.848 | 0.86 |
| S3C | Uncond | 96.363 |

## CelebA
### Diversity 
R1: Gender (Male), R2: Young, R3: Wearing eyeglasses. 

All results are given in percentage of matches
r1-genS: generated images for R1 by Stable diffusion model
r1-genF: generated images for R1 by Flux-LoRA

| | R1 | R2 | R3|
| --- | ----- | ----- | ----- |
| **r1-ori** | 90.93 | 60.42 | 10.48 |
| **r1-genS** | 81 | 78 | 1|
| **r1-genF**| 65 | 52 | 8 |
| **r2-ori** | 28.3 | 96.77 | 2.14 |
| **r2-genS** | 7 | 97 | 0 |
| **r2-genF**| 56 | 64| 7 |
| **r3-ori** | 75.42 | 43.49 | 87.4 |
| **r3-genS** | 55 | 61 | 82 |
| **r3-genF**| 50 | 65 | 36 |


### Flux-LoRA
***Prompt: An attractive young female with arched eyebrows, bags under eyes, big lips, small nose, brown hair, bushy eyebrows, heavy makeup, high cheekbones, mouth slightly open, no beard, pointy nose, smiling, wavy hair, wearing lipstick.***
![flux-lora](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celeba_flux.png)

***Prompt: An old bald man wearing sunglasses.***
![flux-lora](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celeba_flux_1.png)
### VQVAE

***Input image (first row) and reconstructed image(second row) for latent space 3X8X8***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celebahq_vqvae_192.png)

***Input image (first row) and reconstructed image(second row) for latent space 3X16X16***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celebhq_vqvae_768.png)

### Unconditional LDM

***Unconditional LDM with latent space 3X8X8***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celebahq_uncond_192.png)

***Unconditional LDM with latent space 3X16X16***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celebahq_uncond_768.png)

### Text-conditioned LDM

## SGSM
### VQVAE
***Input image (first row) and reconstructed image(second row) for latent space 3X8X28***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/sgsm_vqvae.png)

## MNIST
### VQVAE
***Input image***

![Input sample](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_vqvae_input_samples.png)

***Reconstructed image***

![reconstructed](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_vqvaereconstructed_samples.png)

### Unconditional LDM

![Image of unconditional ldm](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_unconditional_samples.png)

### Text-conditioned LDM
***Texts:***
- Digit 3 with a small area, long length, very thin, right leaned, narrow and high height. (first row of the image grid)
- Digit 0 with a large area, long length, thin, right leaned, wide and high height. (second row of the image grid)
- Digit with extremely right leaned. (third row of the image grid)
- Thick digit. (fourth row of the image grid)
  
![Image of text conditioned ldm](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_text_cond.png)

 
