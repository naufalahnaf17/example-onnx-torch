import torch
from torch import nn
from datasets import get_raw_data,get_loader_data
from simple_neural_network import get_neural_network,get_device
import os

def training(num_epochs=1):
    # Get Raw Train Data and Test Data
    train_data,test_data = get_raw_data()

    # Transform to DataLoader
    train_loader,test_loader = get_loader_data(train_data=train_data,test_data=test_data)
    
    # Get Model
    model = get_neural_network()

    # Get Device
    device = get_device()

    # Setup Criterion (Loss Function) and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    # Training Loop
    
    for epoch in range(num_epochs):
        print(f"Epochs : {epoch + 1}")
        train_model(train_loader=train_loader,optimizer=optimizer,model=model,criterion=criterion,device=device,epoch=epoch)
        test_model(test_loader=test_loader,model=model,device=device,criterion=criterion,epoch=epoch)

    # Save Model
    save_model(model=model)

def train_model(train_loader,optimizer,model,criterion,device,epoch):
    for batch,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images,labels = images.to(device),labels.to(device)
        pred = model(images)

        loss = criterion(pred,labels)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Epoch : {epoch + 1} | Batch : {batch}/{len(train_loader)} Loss : {loss.item():.4f}")

def test_model(test_loader,model,device,criterion,epoch=0):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            test_loss += criterion(pred,labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    correct /= len(test_loader.dataset)
    test_loss /= len(test_loader)
    print(f"Epoch : {epoch + 1} | Accuracy: {(100*correct):.1f}% Loss : {test_loss:.4f}")

def save_model(model):
    if not os.path.exists("./output"):
        os.makedirs("output")

    torch.save(model.state_dict(),"./output/model.pth")
    print("Model saved to /output/model.pth")

def load_model():
    model = get_neural_network()
    model.load_state_dict(torch.load("./output/model.pth"))
    
    train_data,test_data = get_raw_data()
    train_loader,test_loader = get_loader_data(train_data=train_data,test_data=test_data)

    device = get_device()
    criterion = nn.CrossEntropyLoss()

    test_model(test_loader=test_loader,model=model,device=device,criterion=criterion)