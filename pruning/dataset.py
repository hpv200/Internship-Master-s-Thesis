from torch.utils.data import Dataset
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def make_MNIST_dataset (batch):
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
    
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_set)

    return training_loader, validation_loader, test_loader

class ToyDataSet(Dataset):
    """
    class that defines what a data-sample looks like
    In the __init__ you could for example load in the data from file
    and then return specific items in __getitem__
    and return the length in __len__
    """

    def __init__(self, length: int):
        """ loads all stuff relevant for dataset """

        # save the length, usually depends on data-file but here data is generated instead
        self.length = length

        # generate random binary labels
        self.classes = [random.choice([0, 1]) for _ in range(length)]

        # generate data from those labels
        self.data = [np.random.normal(self.classes[i], 0.15, 128) for i in range(length)]

    def __getitem__(self, item_index):
        """ defines how to get one sample """

        class_ = torch.tensor(self.classes[item_index])  # python scalar to torch tensor
        tensor = torch.from_numpy(self.data[item_index])  # numpy array/tensor to torch array/tensor
        return tensor, class_

    def __len__(self):
        """ defines how many samples in an epoch, independently of batch size"""

        return self.length