## Assignment_2: 

**Exercise 1:**\
Implementation of a **Convolutional Neural Network** (CNN) with PyTorch for image classification on the `CIFAR-10 dataset`.

The ConvNet takes 32x32 color images as inputs, has 5 hidden layers with filters sizes: [128, 512, 512, 512, 512] \
and produces a 10-class classification.\
The network architecture consists of five convolutional blocks, each comprising:

- Convolution
- BatchNormalization,
- Pooling,
- ReLU
- Dropout
  
A fully connected layer is used for the classification.

Techniques of geometric and color space **data-augmentations** to improve generalization.

[Assignment outline]
