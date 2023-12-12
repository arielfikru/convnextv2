import torch
import torch.nn as nn
from model import load_convnext_model
from data_loader import load_dataset
from utils import load_model
from config import Config
from sklearn.metrics import classification_report

def evaluate_model(model, dataloader, device):
    """Evaluate the model with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate detailed metrics
    print("Detailed Evaluation Metrics:")
    print(classification_report(all_labels, all_preds, target_names=Config.CLASS_NAMES))

if __name__ == "__main__":
    device = Config.DEVICE
    _, _, test_loader = load_dataset()

    model = load_convnext_model().to(device)
    model = load_model(model, Config.MODEL_SAVE_PATH, device)

    evaluate_model(model, test_loader, device)
