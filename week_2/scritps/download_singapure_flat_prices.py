import kaggle

dataset = 'darrylljk/singapore-hdb-resale-flat-prices-2017-2024'
kaggle.api.dataset_download_files(dataset, path='week_2/data/raw', unzip=True)