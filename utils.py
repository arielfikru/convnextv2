import torch
import time
import copy

def save_model(model, path):
    """
    Save the model state to a file.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path, device=torch.device('cpu')):
    """
    Load the model state from a file.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def time_since(since):
    """
    Calculate elapsed time.
    """
    s = time.time() - since
    m, s = divmod(s, 60)
    return f'{int(m)}m {int(s)}s'

def copy_model(model):
    """
    Return a deep copy of the model.
    """
    return copy.deepcopy(model)

def set_parameter_requires_grad(model, feature_extracting):
    """
    If feature extracting, set all parameters to not require gradients.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
