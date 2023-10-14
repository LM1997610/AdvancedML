# Assignment 2: 
Guidelines [here](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_2/AML_Assignment_2.pdf)

## Exercise 1:

Implementation of a **Convolutional Neural Network** (CNN) with PyTorch for image classification on the `CIFAR-10 dataset`.

The ConvNet takes 32x32 color images as inputs and has 5 hidden layers with filters sizes: [128, 512, 512, 512, 512],\
it produces a 10-class classification.

The network architecture consists of five convolutional blocks, each comprising:

&emsp; Convolution → BatchNormalization → Pooling → ReLU → Dropout
  
A fully connected layer is used for the classification.

Techniques of geometric and color space **data-augmentation** to improve generalization.

[Script](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_2/ex1_convnet.py) and results 
→ Accuracy of the ConvNet on test dataset (1000 test images) : 86.4 %:

![histoy_plot](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/history_plot.png)

Test Accuracy per class :
 
Class | Plane | Car | Bird | Cat | Deer | Dog | Frog | Horse | Boat | Truck
----- | ----- | ----- | ----- |----- |----- |----- |----- | ----- | ----- | ----- 
Test Accuracy | 0.858 | 0.884 | 0.783 | 0.729 | 0.889 | 0.78 | 0.913 | 0.916 | 0.938 | 0.939

<br>

*Visualization of trained filters in the first Convolutional Layer :*
![trained_f](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/filters.png)

## Exercise 2:

Implement **transfer learning** using a pre-trained [VGG-11-bn model] from ImageNet for CIFAR-10 classification.\
Fine-tune the whole network on the CIFAR-10 dataset, starting from the ImageNet initialization.

&ensp; Accuracy of the network on the 1000 test images: 86.9 %
![transfer_lr_1](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/ex2_tuned.png)

Compare this performance to a baseline model where the entire network is trained from scratch without using ImageNet weights.\

