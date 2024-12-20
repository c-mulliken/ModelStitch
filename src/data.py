from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# PARAMS
BATCH_SIZE = 32

MNIST_trainset = datasets.MNIST(root='../data',
                          download=False,
                          train=True,
                          transform=transforms.ToTensor())
MNIST_testset = datasets.MNIST(root='../data',
                          download=False,
                          train=False,
                          transform=transforms.ToTensor())

MNIST_trainloader = DataLoader(MNIST_trainset, batch_size=BATCH_SIZE,
                               shuffle=True)
MNIST_testloader = DataLoader(MNIST_testset, batch_size=BATCH_SIZE,
                               shuffle=False)

