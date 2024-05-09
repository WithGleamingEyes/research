import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        '''First convolutional layer
        3 input channels (the colors)
        16 output channels, 3x3 kernel'''
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  
        '''second convolutional layer
        16 input channels (the colors)
        32 output channels, 3x3 kernel'''
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Input is 32 channels, each 8x8
        self.fc2 = nn.Linear(128, 10)  # Output for 10 classes

    def forward(self, x):
        # Apply convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten for the fully connected layers
        x = torch.flatten(x, 1)
        # use the fully conected layers with ReLU (max(x,0))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

# Load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  
            print(f'[Epoch {epoch + 1}, Iter {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
print('Finished Training')

correct = 0
total = 0
# No gradient computation during testing
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        total += labels.size(0)  # Increment total count
        correct += (predicted == labels).sum().item()  # Increment correct count

print(f'Accuracy on 10,000 test images: {100 * correct / total:.2f}%')
