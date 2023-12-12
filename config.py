import os
import torch

class Config:
    DATASET_DIRECTORY = '/workspace/arknights_op/gallery-dl/danbooru'
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    MODEL_SAVE_PATH = 'arknights.pth'
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WANDB_API_TOKEN = '540d5e6b5446a44f2f5c8269e0da63b4c7ad408c'  # Replace with your actual token or use an environment variable

    @staticmethod
    def get_class_names():
        """
        Get class names from the subfolders of the dataset directory.
        """
        class_names = [folder for folder in os.listdir(Config.DATASET_DIRECTORY)
                       if os.path.isdir(os.path.join(Config.DATASET_DIRECTORY, folder))]
        class_names.sort()
        return class_names

    @staticmethod
    def get_num_classes():
        """
        Count the number of subfolders in the dataset directory.
        """
        return len([name for name in os.listdir(Config.DATASET_DIRECTORY)
                    if os.path.isdir(os.path.join(Config.DATASET_DIRECTORY, name))])

# Set CLASS_NAMES and NUM_CLASSES after the class definition
Config.CLASS_NAMES = Config.get_class_names()
Config.NUM_CLASSES = Config.get_num_classes()
