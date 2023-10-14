
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms

from numpy import linspace
from tqdm .auto import tqdm
import matplotlib.pyplot as plt


#----------------------------------------------------
def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#----------------------------------------------------


#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s \n' %device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 32 * 32 * 3
layer_config = [512, 256]
num_classes = 10
num_epochs = 25
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.99
reg = 0   # 0.001
num_training = 49000
num_validation = 1000
fine_tune = False
pretrained = True

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
data_aug_transforms = [transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomGrayscale(p=0.05), ]

###############################################################################
# Add to data_aug_transforms the best performing data augmentation            #
# strategy and hyper-parameters as found out in Q3.a                          #
###############################################################################

norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) 
                                    # Need to preserve the normalization values of the pre-trained model

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
print(" train_dataset: {}\n val_dataset: {}\n test_dataset: {}".format(len(train_dataset),
                                                                       len(val_dataset),
                                                                       len(test_dataset)))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class VggModel(nn.Module):

    def __init__(self, n_class, fine_tune, pretrained):

        super(VggModel, self).__init__()

        #################################################################################
        # Build the classification network described in Q4 using the                    #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the pretrained flag. You can enable and  #
        # disable training the feature extraction layers based on the fine_tune flag.   #
        #################################################################################

        w_name = "VGG11_Weights.IMAGENET1K_V1" if pretrained == True else None
        pre_trained = models.vgg11(weights = w_name, progress= False)

        if pre_trained.avgpool: del pre_trained.avgpool

        if fine_tune == False: set_parameter_requires_grad(pre_trained, pre_trained.features)

        pre_trained.classifier = nn.Sequential(nn.Flatten(), 
                                            nn.Linear(512, 256), nn.BatchNorm1d(num_features=256), nn.ReLU(),
                                            nn.Linear(256, n_class),nn.BatchNorm1d(num_features=n_class), nn.ReLU())
        
        self.layers = nn.Sequential(*list(pre_trained.children()))

    def forward(self, x):
        
        # Implement the forward pass computations           
        
        out = self.layers(x)
        
        return out

# Initialize the model for this run
model= VggModel(num_classes, fine_tune, pretrained)

if (pretrained==False):
    model.apply(weights_init)

# Print the model we just instantiated
# print(); print(model); print()

##################################################################################
# Only select the required parameters to pass to the optimizer. No need to       #
# update parameters which should be held fixed (conv layers).                    #
##################################################################################

print("\nParams to learn:")

if fine_tune:
    params_to_update = []

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
        
else:
    params_to_update = model.parameters()
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

print(f"\n Pre-Trained = {pretrained}\n Fine-Tune = {fine_tune} \n")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)

loss_train, loss_val = [], []
best_accuracy = None
accuracy_val = []

best_model = type(model)(num_classes, fine_tune, pretrained) # get a new instance

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
        #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, 
        #                                                            i+1, total_step, loss.item()))
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
        
        print('Validataion accuracy is: {} %'.format(accuracy))

        #################################################################################
        # Q2.b Use the early stopping mechanism from previous questions to save         #
        # the model with the best validation accuracy so-far (use best_model).          #
        #################################################################################

        if accuracy == max(accuracy_val):
            best_accuracy = accuracy
            best_model = model
            v_line = epoch

            # Save the model checkpoint
            torch.save(best_model.state_dict(), 'best_model.ckpt')


fig, ax = plt.subplots(1, 2, figsize=(11, 3))
fig.tight_layout(pad=4, w_pad = 6.5)

ax[0].plot(list(range(1, num_epochs+1)), loss_train, 'r', label = 'Train loss')
ax[0].plot(list(range(1, num_epochs+1)), loss_val, 'g', label =' Val loss')
ax[0].set_title('Loss History\n', fontsize=14)
ax[0].set_xlabel('\n Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_ylim(0, 0.008)
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
ax[1].set_xticks(list(range(1, num_epochs+1, 2)))
ax[1].legend(loc ="lower right")
plt.show()

# -------------------------------------------------------------------------------
# Test the model
model.eval() # In test phase, we don't need to compute gradients (for memory efficiency)

#################################################################################
# Use the early stopping mechanism from previous question to load the           #
# weights from the best model so far and perform testing with this model.       #
#################################################################################

best_model = torch.load('best_model.ckpt',  map_location = torch.device('cpu'))
model.load_state_dict(best_model)
# --------------------------------------

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
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

        if total == 1000: break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
