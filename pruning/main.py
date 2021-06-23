from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import sys

from model import FFNET
from dataset import make_MNIST_dataset, ToyDataSet

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def get_accuracy(output, y):
    """ calculates accuracy """
    predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
    correct = y.eq(predictions).sum().item()
    return correct / output.shape[0]
    
def GHS (model, decay):
    reg = 0.0
    if decay:
        for name, param in model.named_parameters():
            if param.requires_grad and len(list(param.size())) == 4 and 'weight' in name and torch.sum(
                    torch.abs(param)) > 0:
                reg += ((torch.sum(torch.sqrt(torch.sum(param ** 2, (0, 2, 3)))) ** 2) + (
                        torch.sum(torch.sqrt(torch.sum(param ** 2, (1, 2, 3)))) ** 2)) / torch.sum(param ** 2)

            elif param.requires_grad and len(list(param.size())) == 2 and 'weight' in name and torch.sum(
                    torch.abs(param)) > 0:
                reg += ((torch.sum(torch.sqrt(torch.sum(param ** 2, 0))) ** 2) + (
                        torch.sum(torch.sqrt(torch.sum(param ** 2, 1))) ** 2)) / torch.sum(param ** 2)

    return reg * decay


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decay = float(list(sys.argv)[2])
    decay_const = float(list(sys.argv)[2])

    # setup
    model = FFNET()
    model.to(device)
    model.set_epochs(int(list(sys.argv)[3]))
    
    loss = nn.CrossEntropyLoss()


    if  list(sys.argv)[1] == 'sample':
        training_loader = DataLoader(
            ToyDataSet(10000),
            batch_size=64,
        )
    elif (list(sys.argv)[1] == 'fMNIST') | (list(sys.argv)[1] == 'fMNIST_Conv6') | (list(sys.argv)[1] == 'fMNIST_Medium'):
        training_loader, validation_loader, test_loader = make_MNIST_dataset(model.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay)

    if list(sys.argv)[1] == 'fMNIST_Conv6':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model.epochs)


    sys.stdout = open("./Pruning/MNIST/Medium/variants/"+ list(sys.argv)[4] +"/outputs_decay" + str(decay) + "_epochs_" + str(model.epochs) + "_pruningRate_"+ list(sys.argv)[4] + ".txt", "a")

    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for epoch in range(model.epochs):
        loss_buffer = []
        acc_buffer = []
        val_loss_buffer = []
        val_acc_buffer = []

        for (batch_inputs, batch_targets) in training_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            # step = step + len(training_loader) * epoch

            # forward pass
            optimizer.zero_grad()
            model.apply_mask()
            prediction = model.forward(batch_inputs.float())
            loss_ = loss.forward(prediction, batch_targets)

            loss_ += GHS(model, decay)

            # backward pass
            loss_.backward()
            optimizer.step()
            model.apply_mask()
            
            # metrics
            with torch.no_grad():
                accuracy = get_accuracy(prediction, batch_targets)
                acc_buffer.append(accuracy)
                loss_buffer.append(loss_.item())

        with torch.no_grad():  # we do not need gradient for validation.
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = loss(val_outputs, val_labels)
                
                val_loss += GHS(model, decay)

                accuracy = get_accuracy(val_outputs, val_labels)

                val_loss_buffer.append(val_loss.item())
                val_acc_buffer.append(accuracy)

        running_loss_history.append(np.mean(loss_buffer))
        running_corrects_history.append(np.mean(acc_buffer))

        val_running_loss_history.append(np.mean(val_loss_buffer))
        val_running_corrects_history.append(np.mean(val_acc_buffer))

        print('\nepoch :', (epoch + 1))
        print('training loss: {:.4f}, acc {:.4f} '.format(np.mean(loss_buffer), np.mean(acc_buffer)))
        print('validation loss: {:.4f}, validation acc {:.4f} '.format(np.mean(val_loss_buffer), np.mean(val_acc_buffer)))
        print('Learning rate', get_lr(optimizer))

        # prune
        if epoch == (69):
            with torch.no_grad():
                model.magnitude_prune_structured(float(list(sys.argv)[4]))
                decay = 0

        if list(sys.argv)[1] == 'fMNIST_Conv6':
            scheduler.step()       
    
    torch.save(model.state_dict(), "./Pruning/MNIST/Medium/variants/Medium_MNIST_weights/medium_weights_MNIST_pruned" + str(list(sys.argv)[5]) + ".pkl")

    
    plt.style.use('ggplot')
    plt.plot(running_corrects_history, label='training accuracy')
    plt.plot(val_running_corrects_history, label='validation accuracy')
    plt.legend()
    plt.savefig("./Pruning/MNIST/Medium/variants/"+ list(sys.argv)[4] +"/accuracy_decay" + str(decay_const) + "_epochs_" + str(model.epochs) + "_pruningRate_"+ list(sys.argv)[4] + ".png")
    plt.clf()


    plt.style.use('ggplot')
    plt.plot(running_loss_history, label='training loss')
    plt.plot(val_running_loss_history, label='validation loss')
    plt.legend()
    plt.savefig("./Pruning/MNIST/Medium/variants/"+ list(sys.argv)[4] +"/loss_decay" + str(decay_const) + "_epochs_" + str(model.epochs) + "_pruningRate_" + list(sys.argv)[4] + ".png")




if __name__ == '__main__':
    main()
