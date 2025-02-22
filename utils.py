"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from torch import utils
from torch.utils import tensorboard 
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


def model_summary(model, 
                  input_size=(1, 3, 224, 224), 
                  verbose=1, col_names=["input_size", "output_size", "num_params", "trainable"], 
                  col_width=20, 
                  row_settings=["var_names"]):
    """
    Generates a summary of the given PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model to summarize.
    input_size (tuple): The size of the input tensor.
    verbose (int): Verbosity level (0 or 1).
    col_names (list): List of column names to display.
    col_width (int): Width of each column.
    row_settings (list): Row settings.

    Returns:
    None
    """
    try:
      summary(model,
            input_size=input_size,
            verbose=verbose,
            col_names=col_names,
            col_width=col_width,
            row_settings=row_settings)
    except:
      print("[ERROR] Make sure you import summary from torchinfo. Can not display the model summary")


# Set seeds
def set_seeds(seed: int=42):

    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def set_device():
    """ Returns the device """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
