import torch
import torch.nn as nn
import torch.nn.transforms as transforms
import torch.nn.functional as F
import torch.optim
#Step 1: Loading and normalizing the dataset

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 0.5 used to convert the pixel values between 0 and 1, into distribution with a mean = 0.5 and std = 0.5

#normalize parameters .normalize(set of Means, set of Standard Deviations, inplace = True/False)
# output[channel] = (input[channel] - mean[channel])/std[channel]

trainset = torchvision.datasets.CIFAR10(root = './data', train=True, download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2) #num_workers used for number of subprocesses for loading the data

testset = torchvision.datasets.CIFAR10(root = './data', train=False, download = True, transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Step 2: Defining a Convolutional Neural Network Layer

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) # in_channels = 3, out_channels = 6, kernel_size = 5*5 square convolution
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x) # output layer
		return x

net = Net()

# to list the parameters of the NN
params = list(net.parameters())
print(len(params))

# Step 3: Defining a loss function and optimizer
# using classification Cross-Entropy and Adam optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


# Step 4: Train the network for three passes over the training dataset

for epoch in range(3):
	run_loss = 0.0
	for i, data in enumerate(trainloader, 0):

		inputs, labels = data
		optimizer.zero_grad() # zero-out the parameter gradients
		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print the statistics
		run_loss += loss.item()
		if i % 2000 == 1999:  # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch+1, i+1, run_loss/2000))
			run_loss = 0.0

print('Finished Training') # print when finished training


# Step 5: Testing the network on test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)

# Output prediction
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# Accuracy on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%' % (100 * correct / total))


# accuracy with these settins : 61%