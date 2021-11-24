import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes


def render_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_some_random_training_images(trainloader, classes, batch_size):
    images, labels = iter(trainloader).next()
    render_image(torchvision.utils.make_grid(images))
    print("labels: ", ' '.join('%5s' %
                               classes[labels[j]] for j in range(batch_size)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_network(trainloader, network, epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


def save_the_nets_params(net, path='./cifar_net.pth'):
    torch.save(net.state_dict(), path)


def load_the_nets_params(path):
    net = Net()
    net.load_state_dict(torch.load(path))
    return net


def sample_one_test_example(network, classes, testloader):
    images, labels = iter(testloader).next()
    print('GroundTruth: ', ' '.join('%5s' %
          classes[labels[j]] for j in range(4)))
    outputs = network(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    render_image(torchvision.utils.make_grid(images))


def print_accuracy(network, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = network(images)
            # The class with the highest energy is
            # what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def print_accuracy_per_class(network, classes, testloader):
    with torch.no_grad():
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for data in testloader:
            images, labels = data
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        for classname, correct_count in correct_pred.items():
            accuracy = 100. * correct_count / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                                 accuracy))


def gpu(network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    network.to(device)
    # Remember that you will have to send the inputs and targets at every step to the GPU too:
    inputs, labels = data[0].to(device), data[1].to(device)
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html



BATCH_SIZE = 4
EPOCHS = 2
trainset, trainloader, testset, testloader, classes = prepare_data(BATCH_SIZE)
network = Net()
train_network(trainloader, network, epochs=EPOCHS)
print_accuracy(network, testloader)
print_accuracy_per_class(network, classes, testloader)
sample_one_test_example(network, classes, testloader)

