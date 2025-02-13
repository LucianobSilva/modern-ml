import os
from kaggle.api.kaggle_api_extended import KaggleApi

DOWNLOAD_PATH = 'week_1/data/raw'

def download_kaggle_dataset():
    # Set up Kaggle API
    print('setting up Kaggle API...')
    api = KaggleApi()
    api.authenticate()

    # Define the competition name
    competition_name = 'neo-bank-non-sub-churn-prediction'

    # Define the download path
    download_path = DOWNLOAD_PATH

    # Ensure the download path exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download all files from the competition
    print('downloading files...')
    api.dataset_download_files(competition_name, path=download_path)

    # Unzip the downloaded files
    print('unzipping files...')
    import zipfile
    with zipfile.ZipFile(os.path.join(download_path, f'{competition_name}.zip'), 'r') as zip_ref:
        zip_ref.extractall(download_path)

    # Remove the zip file
    os.remove(os.path.join(download_path, f'{competition_name}.zip'))

if __name__ == '__main__':
    download_kaggle_dataset()


