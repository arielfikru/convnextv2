import torch
import yaml
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import load_convnext_model
from utils import load_model
from config import Config

def load_class_names(yaml_path='class_names.yaml'):
    """Load class names and their indices from a YAML file."""
    with open(yaml_path, 'r') as file:
        class_name_mapping = yaml.safe_load(file)
    # Inverting the dictionary to map indices to names
    class_names = {v: k for k, v in class_name_mapping.items()}
    return class_names

def preprocess_image(img):
    """Preprocess the image to feed into the model."""
    # Convert to RGB if the image has 4 channels (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def predict(model, img_tensor, device, class_names):
    """Predict the class for an image tensor and return the class name."""
    model.eval()
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class_index = predicted.item()
    return class_names.get(predicted_class_index, "Unknown")

def display_image_with_prediction(img, predicted_class, output_path):
    """Display the image with the predicted class and save to a file."""
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

def process_folder(folder_path, model_path):
    device = Config.DEVICE
    class_names = load_class_names()
    model = load_convnext_model().to(device)
    model = load_model(model, model_path, device)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img_tensor = preprocess_image(img)
            predicted_class = predict(model, img_tensor, device, class_names)

            output_path = os.path.join(folder_path, f"predicted_{filename}")
            display_image_with_prediction(img, predicted_class, output_path)
            print(f"Processed {filename}, saved as {output_path}")

if __name__ == "__main__":
    # Example usage
    folder_path = "/workspace/arknights_op/gallery-dl/zerochan/Blacknight"
    model_path = Config.MODEL_SAVE_PATH
    process_folder(folder_path, model_path)
