import matplotlib.pyplot as plt
import random
from datasets import get_raw_data
from torchvision.transforms import functional as F

train_data,test_data = get_raw_data()

def visualize_single_training_data():
    random_number = random.randint(0,len(train_data))
    image = train_data[random_number][0]
    title = train_data[random_number][1]

    plt.figure()
    plt.imshow(F.to_pil_image(image),cmap=plt.cm.binary)
    plt.title(f"Training Data Title : {title}")
    plt.axis("off")
    plt.show()

def visualize_single_test_data():
    random_number = random.randint(0,len(test_data))
    image = test_data[random_number][0]
    title = test_data[random_number][1]

    plt.figure()
    plt.imshow(F.to_pil_image(image),cmap=plt.cm.binary)
    plt.title(f"Test Data Title : {title}")
    plt.axis("off")
    plt.show()

def visualize_list_training_data():
    plt.figure(figsize=(14,14))
    
    for i in range(25):
        image = train_data[i][0]
        title = train_data[i][1]
        
        plt.subplot(5,5,i+1)
        plt.imshow(F.to_pil_image(image),cmap=plt.cm.binary)
        plt.title(f"Title : {title}")
        plt.axis("off")

    plt.show()

def visualize_list_test_data():
    plt.figure(figsize=(14,14))
    
    for i in range(25):
        image = test_data[i][0]
        title = test_data[i][1]
        
        plt.subplot(5,5,i+1)
        plt.imshow(F.to_pil_image(image),cmap=plt.cm.binary)
        plt.title(f"Title : {title}")
        plt.axis("off")

    plt.show()