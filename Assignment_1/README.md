
# Assignment_1:

## Exercise 1:
Building a Neural Network from scratch → just math and Numpy.\
Simple 2-layer NN and its training algorithm based on **back-propagation** and **stochastic gradient descent**

Benchmark to test the model → image classification task using `CIFAR-10 dataset`

Starting Hyper-parameters: \
&emsp; {'hidden_size': 50, 'learning rate': 0.0001, 'regularization': 0.25, 'iterations': 1000, 'batch_size': 200}
Validation Accuracy: 0.287

![ex1_basic](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_1/images/ex1_basic.png)

Tuning Hyper-parameters: \
&emsp; {'hidden_size': 150, 'learning_rate': 0.004, 'regularization': 0.15, 'iterations': 4000, 'batch_size': 450}
Validation Accuracy: 0.542, &ensp; Test Accuracy:  0.535

   
![ex1_tuned](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_1/images/ex1_tuned.png)
