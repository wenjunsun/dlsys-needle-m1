"""
GPU log:
    Epoch: 0, Acc: 0.38678, Loss: [1.7120664]
    Epoch: 1, Acc: 0.49422, Loss: [1.4014741]
    Epoch: 2, Acc: 0.54014, Loss: [1.2785015]
    Epoch: 3, Acc: 0.57622, Loss: [1.1893052]
    Epoch: 4, Acc: 0.60468, Loss: [1.1160048]
    Epoch: 5, Acc: 0.6258, Loss: [1.0537735]
    Epoch: 6, Acc: 0.64792, Loss: [0.9968544]
    Epoch: 7, Acc: 0.66454, Loss: [0.9435891]
    Epoch: 8, Acc: 0.68322, Loss: [0.89590883]
    Epoch: 9, Acc: 0.69688, Loss: [0.8593632]
    Evaluation Acc: 0.66702, Evaluation Loss: [0.9336795]

M1 log:
    Epoch: 0, Acc: 0.3554, Loss: 1.8095075, Time: 388.40s
    Epoch: 0, Acc: 0.46464, Loss: 1.49002859375, Time: 1002.57s
    Epoch: 0, Acc: 0.51946, Loss: 1.34186625, Time: 1378.20s
    Epoch: 0, Acc: 0.5576, Loss: 1.236773046875, Time: 390.15s
    Epoch: 0, Acc: 0.59026, Loss: 1.1494675, Time: 386.67s
    Epoch: 0, Acc: 0.61794, Loss: 1.075026484375, Time: 388.40s
    Epoch: 0, Acc: 0.64376, Loss: 1.007365703125, Time: 376.15s
    Epoch: 0, Acc: 0.66712, Loss: 0.945820390625, Time: 378.32s
    Epoch: 0, Acc: 0.6903, Loss: 0.89005328125, Time: 385.79s
    Epoch: 0, Acc: 0.71036, Loss: 0.83692109375, Time: 387.82s
"""

import needle as ndl
from needle import backend_ndarray as nd
from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10

device = nd.m1()
train_dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
train_dataloader = ndl.data.DataLoader(dataset=train_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       device=device)
test_dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=False)
test_dataloader = ndl.data.DataLoader(dataset=test_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       device=device)

model = ResNet9(device=device, dtype="float32")
for _ in range(10):
      train_cifar10(model, train_dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                    lr=0.0005, weight_decay=0.001)
      evaluate_cifar10(model, test_dataloader)