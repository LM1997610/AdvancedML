import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from numpy import linspace, rot90
from tqdm.auto import tqdm
from tabulate import tabulate

import matplotlib.pyplot as plt

print(''' 
_ _______________________________________________ _ _
     ______                    _   __     __ 
    / ____/___  ____ _   __   / | / /__  / /_
   / /   / __ \/ __ \ | / /  /  |/ / _ \/ __/
  / /___/ /_/ / / / / |/ /  / /|  /  __/ /_  
  \____/\____/_/ /_/|___/  /_/ |_/\___/\__/  
_ _ ________________________________________________ _ _''')

#-------------------------------------------------------------
def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#-----------------------------------------------------------------------------------


#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\nUsing device: %s\n'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 30
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
# norm_layer = None 
norm_layer = 'BN'


print(f"Hidden_layers: {hidden_size}")
if norm_layer is not None: print(" ... with Batch Normalization\n")


#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
cifar10_class_names = {0: "Plane", 1: "Car", 2: "Bird", 3: "Cat", 4: "Deer", 
                       5: "Dog", 6: "Frog", 7: "Horse", 8: "Boat", 9: "Truck"}

#################################################################################
# Q3.a Choose the right data augmentation transforms with the right             #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################

data_aug_transforms = []

augmentation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop((32,32), padding=3),
                                   transforms.RandomRotation(degrees=15),
                                   transforms.ColorJitter(brightness=0.1, 
                                                          contrast=0.1, saturation=0.1, hue=0.1)])

#data_aug_transforms.append(augmentation) 


norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform = norm_transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=test_transform)

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
print(" train_dataset: {}\n val_dataset: {}\n test_dataset: {}".format(len(train_dataset),
                                                                       len(val_dataset),
                                                                       len(test_dataset)))

#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):

    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):

        super(ConvNet, self).__init__()

        #################################################################################
        # Initialize the modules required to implement the convolutional layer          #
        # described in the exercise.                                                    #
        #                                                                               #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        
        layers = [] 

        hidden_layers =  [128, 512, 512, 512, 512]

        first_layer = nn.Conv2d(in_channels = input_size,  # [32,32,3] --> [32,32,128]
                                out_channels = hidden_layers[0], 
                                kernel_size=3, padding=1)

        pool_layer = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        if norm_layer is None: layers.extend((first_layer, pool_layer, nn.ReLU() ))
        else: layers.extend((first_layer, 
                             nn.BatchNorm2d(hidden_layers[0]), 
                             pool_layer, 
                             nn.ReLU(), 
                             nn.Dropout(0.4)))

        for i in range(len(hidden_layers[1:])):
            
            conv_layer = nn.Conv2d(in_channels = hidden_layers[i],  
                                   out_channels = hidden_layers[i+1], 
                                   kernel_size=3, padding=1)
            
            if norm_layer is None: layers.extend((conv_layer, pool_layer, nn.ReLU()))
            else: layers.extend((conv_layer, 
                                 nn.BatchNorm2d(hidden_layers[i+1]), 
                                 pool_layer, 
                                 nn.ReLU(), 
                                 nn.Dropout(0.3)))
      
        layers.extend((nn.Flatten(), nn.Linear(hidden_layers[-1], num_classes)))

        # Enter the layers into nn.Sequential, so the model may "see" them
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        # Implement the forward pass computations

        out = self.layers(x)

        return out

#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):

    #################################################################################
    # Implement the function to count the number of trainable parameters in the     #
    # input model. This useful to track the capacity of the model you are training  #
    #################################################################################

    model_size = sum(p.numel() for p in model.parameters())

    if disp == True : print(f"\nNumber of parameters: {model_size}\n")

    return model_size

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):

    ##################################################################################
    # Implement the functiont to visualize the weights in the first conv layer       #
    # in the model. Visualize them as a single image of stacked filters.             #
    # You can use matlplotlib.imshow to visualize an image in python                 #
    ##################################################################################
    
    filters = model.layers[0].weight.cpu()

    # normalize data for display
    filters = (filters - filters.min()) / (filters.max() - filters.min())

    gridRows, gridCols = 8, 16
    titles = list(range(1, gridCols + 1))
    ylabels = list(range(1, gridRows + 1))

    fig, axs = plt.subplots(gridRows, gridCols, figsize=(7.5, 3.5), 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axs.flat):

        ax.imshow(filters[i].detach().numpy(), aspect='auto')
        ax.margins(0, 0)

        if i > 111:
            ax.set_xlabel(titles[i-112],  labelpad=15)

        if i in list(range(0, 128, 16)):
            ax.set_ylabel(ylabels[int(i/16)], rotation=0,  labelpad=15)
    plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
def make_single_pred(index, a_dataset, show = True):
        
        model.eval()
        this_img = a_dataset[index][0].reshape(1, 3,32, 32)
        output = model(this_img.to(device))
        _ , predicted = torch.max(output, 1)

        prob = nn.functional.softmax(output[0], dim=0); 
        prob = max(prob).item()
        
        if show == False: return predicted.item(), a_dataset[index][1]

        print(f" True Label: {a_dataset[index][1]} - {cifar10_class_names[a_dataset[index][1]]}")
        print(f" Prediction: {predicted.item()} - {cifar10_class_names[predicted.item()]}, with Prob = {round(prob,4)}")

        fig = plt.figure(figsize=(3, 3))
        this_img = a_dataset[index][0].numpy().T
        this_img = rot90(this_img, 3)
        this_img = ((this_img - this_img.min()) * (1/(this_img.max() - this_img.min()) * 255)).astype('uint8')
        plt.imshow(this_img); plt.gca().axis('off')
        plt.show()
#------------------------------------------------------------------

#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------

model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(); print(model); print()

# Print model size
#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)   # 7682826 con Dropout e BatchNormalization
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
VisualizeFilter(model)
print()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)

loss_train, loss_val = [], []
best_accuracy = None
accuracy_val = []

best_model = type(model)(input_size, hidden_size, num_classes, norm_layer=norm_layer) # get a new instance
best_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer)

for epoch in tqdm(range(num_epochs), position=0, leave=True):

    model.train()

    loss_iter = 0
    for i, (images, labels) in enumerate(train_loader):

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_iter += loss.item()
        
        #if (i+1) % 100 == 0:
        #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, 
        #                                                              total_step, loss.item()))
        
    tqdm.write(f' Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}] - Loss: {loss.item():.4f}', end = "  ")

    loss_train.append(loss_iter/(len(train_loader)*batch_size))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    
    model.eval()
    with torch.no_grad():

        correct = 0
        total = 0
        loss_iter = 0

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            loss_iter += loss.item()
        
        loss_val.append(loss_iter/(len(val_loader)*batch_size))

        accuracy = 100 * correct / total
        accuracy_val.append(accuracy)

        print('Validation accuracy is: {} %'.format(accuracy), end = "\n")

        #################################################################################
        # Q2.b Implement the early stopping mechanism to save the model which has       #
        # the model with the best validation accuracy so-far (use best_model).          #
        #################################################################################

        if accuracy == max(accuracy_val):
            best_accuracy = accuracy
            best_model = model

            # Save the model checkpoint
            torch.save(best_model.state_dict(), 'best_model.ckpt')

# torch.save(model.state_dict(), 'last_model.ckpt')

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
fig.tight_layout(pad=4, w_pad = 6.5)

ax[0].plot(list(range(1, num_epochs+1)), loss_train, 'r', label = 'Train loss')
ax[0].plot(list(range(1, num_epochs+1)), loss_val, 'g', label =' Val loss')
ax[0].set_title('Loss History\n', fontsize=14)
ax[0].set_xlabel('\n Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim(0, 0.010)
ax[0].set_xticks(list(range(1, num_epochs+1, 3)))
ax[0].grid(linewidth=0.4)
ax[0].legend()

ax[1].plot(list(range(1, num_epochs+1)), accuracy_val, 'r', label='Val accuracy')
ax[1].set_title('Classification Accuracy History\n', fontsize=14)
ax[1].set_xlabel('\n Epochs')
ax[1].set_ylabel('Classification accuracy')
ax[1].grid(linewidth=0.4)
ax[1].set_ylim(0, 99); 
ax[1].set_yticks(linspace(0, 100, 9)); 
ax[1].set_xticks(list(range(1, num_epochs+1, 3)))
ax[1].legend(loc ="lower right")

#plt.savefig("history_plot.png")
plt.show()
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()

#################################################################################
# Q2.b Implement the early stopping mechanism to load the weights from the      #
# best model so far and perform testing with this model.                        #
#################################################################################

best_model = torch.load('best_model.ckpt',  map_location=torch.device('cpu'))
    
model.load_state_dict(best_model)

# Compute accuracy on the test set
with torch.no_grad():

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if total == 1000:
            break

    print('\nAccuracy of the network on the {} test images: {} %\n'.format(total, 100 * correct / total))
    #------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------
    correct_diz = {cifar10_class_names[name]:0 for name in cifar10_class_names}
    total_diz = {cifar10_class_names[name]:0 for name in cifar10_class_names}
    sbagliati = []

    for i in tqdm(range(len(test_dataset))):

        pred, true_lab = make_single_pred(i, test_dataset, show = False)

        if pred == true_lab: correct_diz[cifar10_class_names[true_lab]] += 1
        else: sbagliati.append(i)

        total_diz[cifar10_class_names[true_lab]] += 1

    accuracy_per_class = {class_name : correct_diz[class_name]/total_diz[class_name] for class_name in correct_diz}

    print("\n Test Accuracy_per_class :\n")
    rows =  [x.values() for x in [accuracy_per_class]]
    print(tabulate(rows, headers = [accuracy_per_class][0].keys())); print()
#------------------------------------------------------------------------------------


# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')



