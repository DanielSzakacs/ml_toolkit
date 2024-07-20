"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os
import zipfile

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import requests

NUM_WORKERS = os.cpu_count()

def download_data(target_dir_pth: Path, 
                  data_source_pth: str):
  """
    Used to download files (Ex: csv) 
    
    Args:
      target_dir_pth: Path where the file will be downloaded to
      data_source_pth: String which the path to the data (Ex: https://raw.githubusercontent.com/DanielSzakacs/dogImages/main/labels.csv)
  """
  print(f"[INFO] Download data from {data_source_pth}")
  with open(target_dir_pth, "wb") as f: 
      request = requests.get(data_source_pth)
      f.write(request.content)
      print(f"[INFO] Done")


def download_zip_data(source: str,
                      target_path: Path):
   """
    Download and unzip zip files

    Args: 
        source_path: (Path) Ex: /content/drive/MyDrive/Pytorch course/Dog_recognizer/data/train_zip.zip
        target_path: (Path) Where the zip file will be unzipped
        remove_source: (Boolean) Remove the downloaded zip file. 
   """
    # Check is the target dir is exits
   if target_path.is_dir() == False:
      print(f"[INFO] Target directory do not exist...")
      return 
   else: 
      print(f"[INFO] Downloading zip file {source} ...")
      with zipfile.ZipFile(source, "r") as r:
         r.extractall(target_path)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
