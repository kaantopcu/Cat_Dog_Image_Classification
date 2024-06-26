# -*- coding: utf-8 -*-
"""download_data

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17IAHmAqIZGLwc-7hmANyUIotYqkiv7Ge
"""

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset_path = 'data/raw/microsoft-catsvsdogs-dataset.zip'
    if not os.path.exists(dataset_path):
        api.dataset_download_files('shaunthesheep/microsoft-catsvsdogs-dataset', path='data/raw', unzip=False)

    # Extracting the zip files
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw')

def main():
    download_dataset()

if __name__ == "__main__":
    main()