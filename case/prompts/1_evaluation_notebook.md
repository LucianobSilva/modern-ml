Create a Jupyter notebook to evaluate machine learning solutions. Follow these steps:

1. Load the dataset `case/data/processed/test_dataset_with_target.parquet`, which contains the target variable `default`.
2. Iterate through all datasets in the folder `case/data/solutions`. Each dataset can be in either Parquet or CSV format.
3. For each solution dataset, join it with `test_dataset_with_target` on the column `id`.
4. Calculate the log loss for the target variable `default` using the prediction column `probabilities`.
5. Print the log_loss using the file name as the id name of the solution, with the following format:
   1. solution_name, log_loss
   2. João da Silva, 0.01

Ensure that the notebook outputs the log loss for each solution dataset.