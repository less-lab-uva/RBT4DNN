# RBT4DNN
This repository is for the update of the project "RBT4DNN"

Work progress google doc link: [RBT4DNN](https://docs.google.com/document/d/1l_r9Vw-cETf4AvpMhczbUgvMim88Z-tj9eKhTqqIEjk/edit?usp=sharing)

## Baseline Models for Comparison:

Models: (arranged according to column respectively. For example, first column of every image panel is generated from FLUX Dev model. Short description and links of the models can be found here: [RBT4DNN](https://docs.google.com/document/d/1l_r9Vw-cETf4AvpMhczbUgvMim88Z-tj9eKhTqqIEjk/edit?usp=sharing)) 

FLUX Dev, Stable Diffusion XL Base 1.0, Stable Diffusion 3 Medium, Stable Cascade, Image from original dataset (only for CelebA-HQ)

**Prompt:** A man with gray hair
![celeba](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/a_man_with_gray_hair.png)

**Prompt:** Image of digit 1
![mnist](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/Image_of_digit_1.png)

**Prompt:** A vehicle in the lane to the left and within 7 meters
![sgsm](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/A_vehicle_in_the_lane_to_the_left_and_within_7_meters.png)



## CelebA



### Flux-LoRA
***Prompt: An attractive young female with arched eyebrows, bags under eyes, big lips, small nose, brown hair, bushy eyebrows, heavy makeup, high cheekbones, mouth slightly open, no beard, pointy nose, smiling, wavy hair, wearing lipstick.***
![flux-lora](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celeba_flux.png)

***Prompt: An old bald man wearing sunglasses.***
![flux-lora](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/celeba_flux_1.png)

## MNIST
### Flux-LoRA (First row: images from dataset. Second row: images from Flux-LoRA model finetuned on the requirement)

***R1: MNIST hand written digit in white on black background. The digit is a 4 and has very long length.***

![mnist-r1](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r1.png)

***R2: MNIST hand written digit in white on black background. The digit is a 3 and is very thick.***

![mnist-r2](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r2.png)

***R3: MNIST hand written digit in white on black background. The digit is a 7 and is very thick.***

![mnist-r3](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r3.png)

***R4: MNIST hand written digit in white on black background. The digit is a 9 and is very left leaning.***

![mnist-r4](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r4.png)

***R5: MNIST hand written digit in white on black background. The digit is 6 and is very right leaning.***

![mnist-r5](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r5.png)

***R6: MNIST hand written digit in white on black background. The digit is a 0 and has very low height.***

![mnist-r6](https://github.com/nusratdeeptee/RBT4DNN/blob/main/Results/mnist_r6.png)
