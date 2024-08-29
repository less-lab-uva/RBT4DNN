# RBT4DNN
This repository is for the update of the project "RBT4DNN"

Work progress google doc link: [RBT4DNN](https://docs.google.com/document/d/1l_r9Vw-cETf4AvpMhczbUgvMim88Z-tj9eKhTqqIEjk/edit?usp=sharing)
## Metric Score 

| Dataset | Model | cFID | Density | Coverage | Precision | Recall |
| ------ | ----- | ----- | ------ | -------- | ------- | ------- |
| CelebAHQ | Uncond | 
| MNIST | Uncond | 
| S3C | Uncond |

## CelebA
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

## S3C

### Unconditional

![Image of unconditioned ldm](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/s3c_uncond.png)

### Text-conditioned LDM
***Texts:***
- A car is 4 to 7 meters away, in front, to the right of, and in the opposing lane.  A car is 7 to 10 meters away, in front, slightly to the right of, and in the opposing lane. (first row of the image grid)
- Two cars are 7 to 10 meters away, in front, slightly to the right of, and in the opposing lane. (second row of the image grid)

![Image of text conditioned ldm](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/s3c.png)

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

 
