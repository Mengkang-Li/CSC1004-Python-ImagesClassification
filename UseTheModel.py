import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from MainProject import Net
import random

model_dict = torch.load('mnist_cnn.pt')
model = Net()
model.load_state_dict(model_dict)
model.eval()

test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

index = random.randint(0, len(test_dataset)-1)
image, label = test_dataset[index]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)
    prediction = predicted.item()

plt.imshow(image.squeeze(), cmap='gray')
plt.title('Prediction: {}, Label: {}'.format(prediction, label))
plt.show()
