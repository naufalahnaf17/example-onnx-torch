import torch 
import torch.nn as nn
import onnxruntime
from PIL import Image
from torchvision import transforms
import numpy

import onnx
import torch.onnx
from simple_neural_network import get_neural_network
from datasets import get_raw_data,get_loader_data

def export_to_onnx():
    model = get_neural_network()
    model.load_state_dict(torch.load("./output/model.pth"))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 28, 28)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        "./output/mnist_model.onnx", 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Model exported to ONNX successfully")

    onnx_model = onnx.load("./output/mnist_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

def evaluate_models():
    train_data,test_data = get_raw_data()
    train_loader,test_loader = get_loader_data(train_data=train_data,test_data=test_data)

    # Load PyTorch model
    pytorch_model = get_neural_network()
    pytorch_model.load_state_dict(torch.load("./output/model.pth"))
    pytorch_model.eval()

    # Load ONNX model
    onnx_session = onnxruntime.InferenceSession("./output/mnist_model.onnx")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    pytorch_correct = 0
    onnx_correct = 0
    total = 0

    for images, labels in test_loader:
        # PyTorch evaluation
        with torch.no_grad():
            pytorch_outputs = pytorch_model(images)
            _, pytorch_predicted = torch.max(pytorch_outputs, 1)
            pytorch_correct += (pytorch_predicted == labels).sum().item()

        # ONNX evaluation
        onnx_inputs = {onnx_session.get_inputs()[0].name: images.numpy()}
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_predicted = torch.tensor(onnx_outputs[0]).argmax(dim=1)
        onnx_correct += (onnx_predicted == labels).sum().item()

        total += labels.size(0)

    # Calculate and display accuracy
    pytorch_accuracy = 100 * pytorch_correct / total
    onnx_accuracy = 100 * onnx_correct / total

    print(f"PyTorch Model Accuracy: {pytorch_accuracy:.2f}%")
    print(f"ONNX Model Accuracy: {onnx_accuracy:.2f}%")

evaluate_models()