import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from tqdm.auto import tqdm 
import matplotlib.pyplot as plt

print('''
_ ________________________________________________ _ _ _ 
 ______    _____ _                    _     
 | ___ \  |_   _| |                  | |    
 | |_/ /   _| | | |__   ___  _ __ ___| |__      
 |  __/ | | | | | '_ \ / _ \| '__/ __| '_  \  ()
 | |  | |_| | | | | | | (_) | | | (__| | | |
 \_|   \__, \_/ |_| |_|\___/|_|  \___|_| |_|  ()
        __/ |                               
       |___/                                ''')


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}\n')

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
hidden_size = [50]
num_classes = 10
num_epochs = 10
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
train = True

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
norm_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=norm_transform)


#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

print()
print(" train_dataset: {}\n val_dataset: {}\n test_dataset: {}\n".format(len(train_dataset),
                                                                         len(val_dataset),
                                                                         len(test_dataset)))

#======================================================================================
# Q4: Implementing multi-layer perceptron in PyTorch
#======================================================================================
# So far we have implemented a two-layer network using numpy by explicitly
# writing down the forward computation and deriving and implementing the
# equations for backward computation. This process can be tedious to extend to
# large network architectures
#
# Popular deep-learning libraries like PyTorch and Tensorflow allow us to
# quickly implement complicated neural network architectures. They provide
# pre-defined layers which can be used as building blocks to define our
# network. They also enable automatic-differentiation, which allows us to
# define only the forward pass and let the libraries perform back-propagation
# using automatic differentiation.
#
# In this question we will implement a multi-layer perceptron using the PyTorch
# library.  Please complete the code for the MultiLayerPerceptron, training and
# evaluating the model. Once you can train the two layer model, experiment with
# adding more layers and report your observations
#--------------------------------------------------------------------------------------

#-------------------------------------------------
# Fully connected neural network with one hidden layer
#-------------------------------------------------


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_layers, num_classes):

        super(MultiLayerPerceptron, self).__init__()
        
        #################################################################################
        # Initialize the modules required to implement the mlp with the layer           #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        # Make use of linear and relu layers from the torch.nn module                   #
        #################################################################################
        
        layers = [] #Use the layers list to store a variable number of layers
        
        for i in range(len(hidden_layers)):
    
            if i == 0: # input layer
                this_layer = nn.Linear(input_size, hidden_size[i]); 
                layers.extend((nn.Flatten(), this_layer, nn.ReLU()))
    
            else: # hidden layers
                this_layer = nn.Linear(hidden_size[i-1], hidden_size[i]); 
                layers.extend((this_layer, nn.ReLU()))
    
        last_layer = nn.Linear(hidden_size[-1], num_classes)
        layers.append(last_layer)

        # Enter the layers into nn.Sequential, so the model may "see" them
        self.layers = nn.Sequential(*layers)  # Note the use of * in front of layers

    def forward(self, x):

        #################################################################################
        # # Implement the forward pass computations                                     #
        # Note that you do not need to use the softmax operation at the end.            #
        # Softmax is only required for the loss computation and the criterion used below#
        # nn.CrossEntropyLoss() already integrates the softmax and the log loss together#
        #################################################################################

        out = self.layers(x)

        return out

model = MultiLayerPerceptron(input_size, hidden_size, num_classes).to(device)

''' 
print()  # Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())'''


if train:
    model.apply(weights_init)
    model.train() #set dropout and batch normalization layers to training mode

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    total_step = len(train_loader)

    # track records 
    loss_history, loss_history_val = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in tqdm(range(num_epochs), position=0, leave=True):

        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            ######################################################################################
            # Implement the training code                                                        #
            # 1. Pass the images to the model                                                    #
            # 2. Compute the loss using the output and the labels.                               #
            # 3. Compute gradients and update the model using the optimizer                      #
            # Use examples in https://pytorch.org/tutorials/beginner/pytorch_with_examples.html  #
            ######################################################################################

            output = model(images)
            _ , pred = torch.max(output, 1)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

            #if (i+1) % 100 == 0:
            #    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 
            #                                                             total_step, loss.item()))
        
        # Code to update the lr
        lr *= learning_rate_decay
        update_lr(optimizer, lr)

        with torch.no_grad():

            correct = 0
            total = 0

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                ####################################################
                #  Implement the evaluation code                   #
                # 1. Pass the images to the model                  #
                # 2. Get the most confident predicted class        #
                ####################################################

                output= model(images)
                _ , predicted = torch.max(output, 1)
                val_loss = criterion(output, labels)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            #print('Validataion accuracy is: {} %'.format(100 * correct / total))
            tqdm.write(f' Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}, val_accuracy = {(100 * correct / total)} %')

        loss_history.append(round(loss.item(), 4))
        loss_history_val.append(round(val_loss.item(), 4))

        train_acc_history.append(100 * correct_train / total_train)
        val_acc_history.append(100 * correct / total)
        
    
    stats = {"loss_history_train": loss_history, "loss_history_val": loss_history_val,
            "train_acc_history": train_acc_history, "val_acc_history": val_acc_history}
    

    print()
    #----------------------------------------------------------------------
    # Plot the loss function and train / validation accuracies
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.tight_layout(pad=4.0, w_pad=6)

    r = list(range(1, num_epochs+1))
    ax[0].plot(r , stats['loss_history_train'], label = "train")
    ax[0].plot(r, stats['loss_history_val'], label='validation')
    ax[0].set_title('Loss History\n', fontsize=14)
    ax[0].set_xlabel('\n Epochs'); ax[0].set_ylabel('Loss')
    ax[0].set_ylim(0.9, 2.2); ax[0].set_xticks(r)
    ax[0].grid(linewidth=0.4); ax[0].legend()

    ax[1].plot(r, stats['train_acc_history'], label='train')
    ax[1].plot(r, stats['val_acc_history'], label='validation')
    ax[1].set_title('Classification Accuracy History\n', fontsize=14)
    ax[1].set_xlabel('\n Epochs'); ax[1].set_ylabel('Classification accuracy')
    ax[1].grid(linewidth=0.4); ax[1].set_ylim(0, 100); ax[1].legend()
    ax[1].set_xticks(r)
    plt.show()
    #plt.savefig("baseline.png") #----------------------------------------------

    ##################################################################################
    # TODO: Now that you can train a simple two-layer MLP using above code, you can  #
    # easily experiment with adding more layers and different layer configurations   #
    # and let the pytorch library handle computing the gradients                     #
    #                                                                                #
    # Experiment with different number of layers (at least from 2 to 5 layers) and   #
    # record the final validation accuracies Report your observations on how adding  #
    # more layers to the MLP affects its behavior. Try to improve the model          #
    # configuration using the validation performance as the guidance. You can        #
    # experiment with different activation layers available in torch.nn, adding      #
    # dropout layers, if you are interested. Use the best model on the validation    #
    # set, to evaluate the performance on the test set once and report it            #
    ##################################################################################

    # Save the model checkpoint
    # torch.save(model.state_dict(), 'model.ckpt')

else:
    # Run the test code once you have your by setting train flag to false
    # and loading the best model

    best_model = None
    best_model = torch.load('model.ckpt',  map_location=torch.device('cpu'))
    
    model.load_state_dict(best_model)
    
    # Test the model
    model.eval() #set dropout and batch normalization layers to evaluation mode
    
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():

        correct = 0
        total = 0

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            ####################################################
            # Implement the evaluation code                    #
            # 1. Pass the images to the model                  #
            # 2. Get the most confident predicted class        #
            ####################################################

            output= model(images)
            _ , predicted = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if total == 1000:
                break

        print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

