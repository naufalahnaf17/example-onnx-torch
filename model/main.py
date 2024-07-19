from training import training,load_model
from visualize import visualize_single_training_data,visualize_single_test_data,visualize_list_training_data,visualize_list_test_data
from export import export_to_onnx,evaluate_models

def __main__():
    # Visualize data for better understanding
    visualize_single_training_data()
    visualize_single_test_data()
    visualize_list_training_data()
    visualize_list_test_data()

    # Training Model
    training(num_epochs=10)

    # Load Model To Evaluate
    load_model()

    # Export to ONNX
    export_to_onnx()

    # Evaluate Model Pytorch and ONNX
    evaluate_models()


if __name__ == "__main__":
    __main__()