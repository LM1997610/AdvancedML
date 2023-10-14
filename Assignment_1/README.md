
# Assignment_1:
Guidelines [here](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_1/AML_Assignment_1.pdf)
## Exercise 1:
Building a Neural Network from scratch → just math and Numpy.\
Simple 2-layer NN and its training algorithm based on **back-propagation** and **stochastic gradient descent**.\
Implemented as Class object in [two_layernet.py](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_1/two_layernet.py)

Benchmark to test the model → image classification task using `CIFAR-10 dataset`

- Provided Hyper-parameters: \
&emsp; {'hidden_size': 50, 'learning rate': 1e-4, 'regularization': 0.25, 'iterations': 1000, 'batch_size': 200}

&emsp; *Validation Accuracy: 0.287*
![ex1_basic](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_1/images/ex1_basic.png)

- Tuning Hyper-parameters: \
&emsp; {'hidden_size': 150, 'learning_rate': 4e-3, 'regularization': 0.15, 'iterations': 4000, 'batch_size': 450}

&emsp; *Validation Accuracy: 0.542, &ensp; Test Accuracy:  0.535*
![ex1_tuned](https://github.com/LM1997610/AdavancedML/blob/main/Assignment_1/images/ex1_tuned.png)


## Exercise 2:

Multi-Layer Perceptron Network with `PyTorch` to implement again the classification we did before, training on the same CIFAR-10 dataset.\
**PyThorch** simplifies neural network experimentation, extending easily a two-layer network to a three or four-layered one.

Reference code in [ex2_PyTorch.py](https://nbviewer.org/github/LM1997610/AdavancedML/blob/main/Assignment_1/ex2_PyTorch.py)



