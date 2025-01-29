import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_lending_club_dataset(destination_dir):
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Define the dataset name
    dataset_name = 'marcusos/lending-club-clean'
    
    # Download the dataset
    print('Downloading the dataset...')
    api.dataset_download_files(dataset_name, path=destination_dir, unzip=True)
    
    print(f"Dataset downloaded and extracted to {destination_dir}")

if __name__ == "__main__":
    # Specify the directory to save the dataset
    destination_dir = './case/data/raw'
    
    # Create the directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Download the dataset
    download_lending_club_dataset(destination_dir)
