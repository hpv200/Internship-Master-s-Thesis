import torch.nn as nn
import sys
import numpy as np

from PrunableModel import PrunableModel


class FFNET(PrunableModel):

    def __init__(self, **kwargs):
        super(FFNET, self).__init__(**kwargs)
        self.name = sys.argv[1]

        if self.name == 'sample':
            self.layers = nn.Sequential(
                nn.Linear(128, 76),
                nn.ReLU(),
                nn.Linear(76, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 2)
            )
        elif self.name == 'fMNIST':
            self.lr = 0.001
            self.epochs = 15
            self.batch_size = 500
            self.weight_decay = 1e-5

            self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.classifier = nn.Sequential(
                nn.Dropout2d(0.3),
                nn.Linear(in_features=64 * 7 * 7, out_features=64),
                nn.ReLU(),
                nn.Dropout2d(0.3),
                nn.Linear(in_features=64, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=10)
            )

        elif self.name == 'fMNIST_Conv6':
            self.lr = 0.01
            self.epochs = 10
            self.batch_size = 500
            self.weight_decay = 1e-5

            self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=True),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
            )

            self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

            self.classifier = nn.Sequential(

                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(256 * np.prod((3, 3)), 256, bias=True),
                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),

                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(256, 256, bias=True),
                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),

                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(256, 10, bias=True),
            )
            
        elif self.name == 'fMNIST_Medium':
            self.lr = 0.0001
            self.epochs = 100
            self.batch_size = 200
            self.weight_decay = 1e-7
            
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.classifier = nn.Sequential(
                nn.Dropout2d(0.3),
                nn.Linear(in_features=128 * 7 * 7, out_features=128),
                nn.ReLU(),
                nn.Dropout2d(0.3),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=10)
            )

        self._post_init()
    
    def set_epochs(self, n):
        self.epochs = n

    def forward(self, x):
        if self.name == 'sample':
            return self.layers.forward(x)
            
        elif (self.name == 'fMNIST') | (self.name == 'fMNIST_Medium'):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
            
        elif self.name == 'fMNIST_Conv6':
            x = self.features.forward(x)
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            return x

        raise ValueError('Model not defined')
        return
