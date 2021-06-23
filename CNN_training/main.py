import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split

import sys

import matplotlib.pyplot as plt

from models import FashionCNN, Conv6, MediumCNN

def get_accuracy(output, y):
    """ calculates accuracy """
    predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
    correct = y.eq(predictions).sum().item()
    return correct / output.shape[0]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    name = sys.argv[5]

    if name == 'Medium':
        model = MediumCNN()
    elif name == 'Conv':
        model = Conv6()
    elif name == 'Base':
        model = FashionCNN()
        
    
    model.to(device)
    
    transform_train = transforms.Compose(
    [transforms.Resize((28, 28)),  # resises the image so it can be perfect for our model.
     transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
     transforms.RandomRotation(10),  # Rotates the image to a specified angel
     transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
     # Performs actions like zooms, change shear angles.
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
     transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
     transforms.Normalize(([0.5]), ([0.5]))  # Normalize all the images
     ])

    train_set = torchvision.datasets.FashionMNIST("./CNNs/mnist_conv6/data", download=True, train=True, transform=transform_train)
    
    test_set = torchvision.datasets.FashionMNIST("./CNNs/mnist_conv6/data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(([0.5]), ([0.5]))]))
    
    validation_set, test_set = random_split(test_set, [int(len(test_set)/2),int(len(test_set)/2)], generator=torch.Generator().manual_seed(27))
    
    batch = int(sys.argv[1])
    epochs = int(sys.argv[2])
    weight_decay = float(sys.argv[3])
    lr = float(sys.argv[4])
    
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_set)


    criterion = nn.CrossEntropyLoss()  # same as categorical_crossentropy loss used in Keras models which runs on Tensorflow
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # fine tuned the lr
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    sys.stdout = open("./CNNs/mnist_conv6/temp_" + name +"/outputs" + '_lr'+ str(lr)+ '_weidecay_' + str(weight_decay) +'_batch_' + str(batch) + "_epoch_" + str(epochs) + ".txt", "a")

    for e in range(epochs):  # training our model, put input according to every batch.

        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for inputs, labels in training_loader:
            inputs = inputs.to(device)  # input to device as our model is running in mentioned device.
            labels = labels.to(device)
            outputs = model(inputs)  # every batch of 100 images are put as an input.

            loss = criterion(outputs, labels)  # Calc loss after each batch i/p by comparing it to actual labels.

            optimizer.zero_grad()  # setting the initial gradient to 0
            loss.backward()  # backpropagating the loss
            optimizer.step()  # updating the weights and bias values for every single step.

            _, preds = torch.max(outputs, 1)  # taking the highest value of prediction.
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)  # calculating te accuracy by taking the sum of all the correct predictions in a batch.
        else:
            with torch.no_grad():  # we do not need gradient for validation.
                for val_inputs, val_labels in validation_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data)

        epoch_loss = running_loss / len(training_loader)  # loss per epoch
        epoch_acc = running_corrects.float() / (len(training_loader) * training_loader.batch_size)  # accuracy per epoch
        running_loss_history.append(epoch_loss)  # appending for displaying
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_acc = val_running_corrects.float() / (len(validation_loader) * validation_loader.batch_size)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        print('\nepoch :', (e + 1))
        print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
        print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))
        print('Learning rate', get_lr(optimizer))
        
        torch.save(model.state_dict(), "./CNNs/mnist_conv6/" + name +"_MNIST_weights/weights_MNIST_" + name + str(e) + ".pkl")

        if name == 'Conv':
            scheduler.step()


    plt.style.use('ggplot')
    plt.plot(running_loss_history, label='training loss')
    plt.plot(val_running_loss_history, label='validation loss')
    plt.legend()
    plt.savefig('./CNNs/mnist_conv6/temp_' + name +'/loss' + '_lr'+ str(lr)+ '_weidecay_' + str(weight_decay) +'_batch_' + str(batch) + "_epoch_" + str(epochs) +'.png')
    plt.show()
    plt.clf()

    plt.style.use('ggplot')
    plt.plot(running_corrects_history, label='training accuracy')
    plt.plot(val_running_corrects_history, label='validation accuracy')
    plt.legend()
    plt.savefig('./CNNs/mnist_conv6/temp_' + name +'/accuracy' + '_lr'+ str(lr)+ '_weidecay_' + str(weight_decay) +'_batch_' + str(batch) + "_epoch_" + str(epochs)+'.png')
    plt.show()

    sys.stdout.close()
if __name__ == '__main__':
    main()