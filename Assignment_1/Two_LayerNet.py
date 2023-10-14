
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

#try: xrange     # Python 2
#except NameError: 
xrange = range   # Python 3

class TwoLayerNet(object):
 
    """A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer. In other words, the network has the following architecture:

    input -> fully connected layer -> ReLU -> fully connected layer -> softmax

    The outputs of the second fully-connected layer are the scores for each class."""

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        
        """Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C."""
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, y=None, reg=0.0):
        
        """Compute the loss and gradients for a two-layer fully connected NeuralNetwork.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params"""
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1'] # shapes (input_size, 10) -- (10,)
        W2, b2 = self.params['W2'], self.params['b2'] # shapes (10, 3) -- (3, )
        N, D = X.shape

        # Compute the forward pass
        scores = 0.0
        
        ##############################################################################
        # Perform the forward pass, computing the class probabilities for the input. #
        # Store the result in the scores variable, which should be an array          #
        # of shape (N, C).                                                           #
        ##############################################################################

        zeta_1 = np.dot(X, W1) + b1 
        activation_1 = np.maximum(0, zeta_1)

        zeta_2 = np.dot(activation_1, W2) + b2

        softmax = np.exp(zeta_2)
        scores = np.divide(softmax.T, np.sum(softmax, axis=1)).T
      
        # If the targets are not given then jump out, we're done
        if y is None: return scores

        # Compute the loss
        loss = 0.0
        
        #############################################################################
        # Finish the forward pass, and compute the loss. This should include both   #
        # the data loss and L2 regularization for W1 and W2. Store the result  in   #
        # the variable loss, which should be a scalar. Use the Softmax              #
        # classifier loss.                                                          #
        #############################################################################
        
        # Implement the loss for the softmax output layer
        
        one_hot = np.zeros((y.size, y.max() + 1))
        one_hot[np.arange(y.size), y] = 1
        
        this_loss = - np.sum( one_hot * np.log(scores)) / N
        regularisation = np.sum((W1)**2) + np.sum((W2)**2)
        loss = this_loss + reg  * regularisation

        # Backward pass: compute gradients
        grads = {}

        ##############################################################################
        # Implement the backward pass, computing the derivatives of the weights      #
        # and biases. Store the results in the grads dictionary. For example,        #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size  #
        ##############################################################################
        
        last_layer_error = scores-one_hot

        grads['W2'] = 1/N * np.dot(activation_1.T, last_layer_error) + 2*reg*W2
        grads["b2"] = 1/N * np.sum(last_layer_error, axis = 0)

        
        delta_z1 = np.dot(W2, last_layer_error.T).T * (zeta_1 > 0) * 1

        grads['W1']= 1/N * np.dot(X.T, delta_z1) + 2*reg*W1
        grads['b1'] = 1/N * np.sum(delta_z1, axis = 0)

        return loss, grads


    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        
        """Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization """
        
        num_train = X.shape[0]
        iterations_per_epoch = max( int(num_train // batch_size), 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in tqdm(range(num_iters),  position=0, leave=True):
            
            #########################################################################
            # Create a random minibatch of training data and labels, storing        #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            batch_indicies = np.random.choice(num_train, batch_size)
            
            X_batch = X[batch_indicies]
            y_batch = y[batch_indicies]
            #pass

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # Use the gradients in the grads dictionary to update the               #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            
            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["b1"] -= learning_rate * grads["b1"]
            self.params["W2"] -= learning_rate * grads["W2"]
            self.params["b2"] -= learning_rate * grads["b2"]
            #pass

            if verbose and it % 100 == 0:
                tqdm.write(' iteration %d / %d: loss %f' % (it, num_iters, loss))

            # At every epoch check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                
              train_acc = (self.predict(X_batch) == y_batch).mean()
              val_acc = (self.predict(X_val) == y_val).mean()
              train_acc_history.append(train_acc)
              val_acc_history.append(val_acc)

              # Decay learning rate
              learning_rate *= learning_rate_decay


        return {'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history}



    def predict(self, X, single = None):
        
        """ Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C."""

        zeta_1 = np.dot(X, self.params["W1"]) + self.params["b1"]
        activation_2 = np.maximum(0, zeta_1)

        zeta_2 = np.dot(activation_2, self.params["W2"]) + self.params["b2"]
        softmax = np.exp(zeta_2)
        

        if single == True: 
            scores = np.divide(softmax.T, np.sum(softmax, axis = 0)).T
            y_pred = np.argmax(scores.T, axis = 0) 
            return y_pred, np.max(scores)

        scores = np.divide(softmax.T, np.sum(softmax, axis = 1)).T
        y_pred = np.argmax(scores.T, axis = 0) 

        return y_pred
    
    def make_single_pred(self, index, x_test, y_test, show = True):

        cifar10_class_names = {0: "Plane", 1: "Car", 2: "Bird", 3: "Cat", 4: "Deer", 
                               5: "Dog", 6: "Frog", 7: "Horse", 8: "Boat", 9: "Truck"}
        
        this_img = x_test[index].reshape(32,32,3)
        this_img = ((this_img - this_img.min()) * (1/(this_img.max() - this_img.min()) * 255)).astype('uint8')
        
        y_pred, prob = self.predict(x_test[index], single = True)
        
        print(f" True Label: {y_test[index]} - {cifar10_class_names[y_test[index]]}" )
        print(f" Prediction: {y_pred} - {cifar10_class_names[y_pred]}, with Prob = {round(prob,3)}")
        
        if show == True: 
            fig = plt.figure(figsize=(3, 3))
            plt.gca().axis('off')
            plt.imshow(this_img)
    
    
    def do_my_plot(self, statistics : dict, step = 3):
    
      # Plot the loss function and train / validation accuracies

      fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
      fig.tight_layout(pad=4, w_pad= 6)

      ax[0].plot(statistics['loss_history'], label ='train')
      ax[0].set_title('Loss History\n', fontsize=14)
      ax[0].set_xlabel('\n Iterations'); ax[0].set_ylabel('\n Loss')
      ax[0].grid(linewidth=0.4); ax[0].legend()

      r = list(range(1, len(statistics['train_acc_history']) +1))
      ax[1].plot( r, statistics['train_acc_history'], label ='train')
      ax[1].plot( r, statistics['val_acc_history'], label='validation')

      ax[1].set_title('Classification Accuracy History\n', fontsize=14)
      ax[1].set_xlabel('\n Epochs'); ax[1].set_ylabel('Classification accuracy')
      
      ax[1].set_xticks(list(range(1, len(statistics['train_acc_history'])+1, step)))
      ax[1].set_yticks(np.linspace(0, 1, 9)); ax[1].set_ylim(0, 0.999)
      ax[1].grid(linewidth=0.4); ax[1].legend(loc="best")
      plt.show()


