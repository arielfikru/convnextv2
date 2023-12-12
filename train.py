import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import wandb
from tqdm import tqdm
from model import load_convnext_model
from data_loader import load_dataset
from utils import save_model
from config import Config

device = Config.DEVICE
print(f"Using device: {device}")

# Initialize wandb
os.environ["WANDB_API_KEY"] = Config.WANDB_API_TOKEN
wandb_project_name = Config.MODEL_SAVE_PATH.split('.')[0]  # Use the model name as the project name
wandb.init(project=wandb_project_name)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model = model.to(device)

    best_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            data_loader = train_loader if phase == 'train' else val_loader
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                progress_bar.set_postfix(loss=running_loss/len(data_loader.dataset), 
                                         accuracy=(running_corrects.double()/len(data_loader.dataset)).item())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log metrics to wandb
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_acc, "epoch": epoch})

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # Save the best model
    model.load_state_dict(best_model_wts)
    save_model(model, Config.MODEL_SAVE_PATH)

    # Summary
    print(f"\nFinish Fine-tuning: Best Val Acc: {best_acc:.4f}")

    # Save class names and numbers to YAML file
    class_info = {name: idx for idx, name in enumerate(Config.CLASS_NAMES)}
    with open('class_names.yaml', 'w') as file:
        yaml.dump(class_info, file)

    return model

if __name__ == "__main__":
    train_loader, val_loader, _ = load_dataset()
    model = load_convnext_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, Config.NUM_EPOCHS)
