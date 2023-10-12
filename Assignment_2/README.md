## Assignment_2: 

**Exercise 1:**\
Implementation of a **Convolutional Neural Network** (CNN) with PyTorch for image classification on the `CIFAR-10 dataset`.

The ConvNet takes 32x32 color images as inputs, has 5 hidden layers with filters sizes: [128, 512, 512, 512, 512] \
and produces a 10-class classification.

The network architecture consists of five convolutional blocks, each comprising:

&emsp; Convolution → BatchNormalization → Pooling → ReLU → Dropout
  
A fully connected layer is used for the classification.

Techniques of geometric and color space **data-augmentation** to improve generalization.

Accuracy of the ConvNet on test dataset (1000 test images) : 85.2 %


![alt text](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/history_plot.png)

[Assignment outline](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_2/AML_Assignment_2_ConvNets.pdf)
