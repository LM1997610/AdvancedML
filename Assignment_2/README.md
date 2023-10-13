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
→ Accuracy of the ConvNet on test dataset (1000 test images) : 85.9 %:

![histoy_plot](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/history_plot.png)

Test Accuracy_per_class :

 'Plane': 0.894
 'Car': 0.934
 'Bird': 0.771
 'Cat': 0.701
 'Deer': 0.865
 'Dog': 0.754
 'Frog': 0.926
 'Horse': 0.882
 'Boat': 0.932
 'Truck': 0.894
 
Class | #Plane | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

*Visualization of trained filters in the first Convolutional Layer :*
![trained_f](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_2/images/filters.png)

## Exercise 2:

