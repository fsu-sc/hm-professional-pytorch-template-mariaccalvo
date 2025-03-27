#  2. Implement Model Architecture 

import torch
import torch.nn as nn
from torch.optim import Adam

# activation functions
activations = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'linear': nn.Identity  
}

# base model definition
class BaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def train_model(self, dataloader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            self.train() 
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    def evaluate_model(self, dataloader, criterion):
        self.eval()  
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        print(f"Validation Loss: {running_loss/len(dataloader)}")

# dense model definition
class DenseModel(BaseModel):
    def __init__(self, input_size, output_size, hidden_layers=1, neurons_per_layer=10, activation_hidden='relu', activation_output='linear'):
        super().__init__(input_size, output_size)

        # check activation function
        if activation_hidden not in activations:
            raise ValueError(f"Unknown activation function: {activation_hidden}. Activation options only are relu, sigmoid, tanh, and linear.")

        # list of layers
        layers = []

        # input layer
        layers.append(nn.Linear(input_size, neurons_per_layer))

        # hidden layers
        for _ in range(hidden_layers):
            layers.append(activations[activation_hidden]())  
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))

        # output layer
        layers.append(activations[activation_output]())   
        layers.append(nn.Linear(neurons_per_layer, output_size))

        # combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
