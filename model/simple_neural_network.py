import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_neural_network():
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # input = 1 * 28 * 28 (1 channel color , 28 width , 28 height) and output = 512
            self.ln1 = nn.Linear(1 * 28 * 28 , 512)
            # input = 512 and output = 512
            self.ln2 = nn.Linear(512 , 512)
            # input = 512 and output = 10 (number class MNIST dataset is 0-9 so total number length is 10)
            self.ln3 = nn.Linear(512 , 512)

        def forward(self,x):
            # Change tensor to 1d array from [1,1,28,28] -> [1, 1 * 28 * 28] or [1,784]
            x = x.view(x.size(0) , -1)
            # # Input 1d array to linear with relu functional
            x = F.relu(self.ln1(x))
            x = F.relu(self.ln2(x))
            x = self.ln3(x)
            return x
        
    model = NeuralNetwork().to(device)
    return model

def test_neural_network():
    model = get_neural_network()
    input_dummy = torch.rand(1,1,28,28).to(device)
    output = model(input_dummy)
    print(output)

def get_device():
    using_device = device
    return using_device