import gradio as gr
import torch
import yaml
from PIL import Image
from torchvision import transforms
from model import load_convnext_model
from utils import load_model
from config import Config

def load_class_names(yaml_path='class_names.yaml'):
    """Load class names from a YAML file."""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def preprocess_image(img):
    """Preprocess the image to feed into the model."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def predict_image(img, model_path):
    """Predict the class for an image using the specified model."""
    device = Config.DEVICE
    class_names = load_class_names()
    model = load_convnext_model().to(device)
    model = load_model(model, model_path, device)

    img_tensor = preprocess_image(img)
    model.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class_index = predicted.item()
    if predicted_class_index in class_names:
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = "Unknown"
    
    return predicted_class_index, predicted_class_name

def main():
    model_path = gr.inputs.Textbox(label="Path to Model File")
    image_input = gr.inputs.Image(label="Upload Image")
    outputs = [gr.outputs.Textbox(label="Predicted Class Index"), 
               gr.outputs.Textbox(label="Predicted Class Name")]

    gr.Interface(fn=predict_image, inputs=[image_input, model_path], outputs=outputs, 
                 title="Image Classification with ConvNeXT V2").launch()

if __name__ == "__main__":
    main()
